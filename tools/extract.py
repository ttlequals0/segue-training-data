"""Extract training examples from a MinusPod database copy.

Reconstructs the per-window detection prompts with MinusPod's own
create_windows/format_window_prompt (the DB stores only a placeholder for
first_pass_prompt) and pairs each window with the final ad markers that
intersect it.

Usage:
    uv run python tools/extract.py --db .local/minuspod.db [--limit 25]
"""
import argparse
import datetime
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    EXAMPLES_DIR, EXTRACTOR_VERSION, load_excluded_feeds, load_holdout,
    setup_minuspod_path, store_prompt,
)

WINDOW_SIZE_DEFAULT = 600.0
WINDOW_OVERLAP_DEFAULT = 180.0
MIN_AD_OVERLAP_SECONDS = 1.0


def read_window_settings(conn):
    size, overlap = WINDOW_SIZE_DEFAULT, WINDOW_OVERLAP_DEFAULT
    rows = conn.execute(
        "SELECT key, value FROM settings WHERE key IN "
        "('window_size_seconds', 'window_overlap_seconds')").fetchall()
    for key, value in rows:
        try:
            if key == 'window_size_seconds':
                size = float(value)
            else:
                overlap = float(value)
        except (TypeError, ValueError):
            pass
    return size, overlap


def build_description_section(podcast_description, episode_description):
    # Mirrors AdDetector.detect_ads; known-pattern and positional hints are
    # omitted in this slice (time-varying DB state).
    section = ""
    if podcast_description:
        section = f"Podcast Description:\n{podcast_description}\n\n"
    if episode_description:
        section += (
            "Episode Description (this describes the actual content topics "
            "discussed; it may also list episode sponsors):\n"
            f"{episode_description}\n")
    return section


def trailing_words(text, n=5):
    words = text.split()
    return " ".join(words[-n:]) if words else ""


def window_completion(markers, window, segments):
    """Ads intersecting this window, clipped to its bounds, sorted by start."""
    out = []
    for m in markers:
        s = max(float(m['start']), window['start'])
        e = min(float(m['end']), window['end'])
        if e - s < MIN_AD_OVERLAP_SECONDS:
            continue
        entry = {
            'start': round(s, 2),
            'end': round(e, 2),
            'confidence': float(m.get('confidence', 0.9)),
            'category': m['category'],
            'reason': str(m.get('reason', '')),
            'end_text': str(m.get('end_text', '')),
        }
        if float(m['end']) > window['end']:
            # Clipped: teacher end_text describes audio past this window.
            inside = [seg for seg in segments
                      if seg['end'] <= window['end'] and seg['start'] >= s]
            if inside:
                entry['end_text'] = trailing_words(inside[-1]['text'])
        out.append(entry)
    return sorted(out, key=lambda a: a['start'])


def usable_markers(markers):
    """Cut markers only; None when the episode's labels are unusable."""
    kept = []
    for m in markers:
        if not m.get('was_cut', True) or m.get('held_for_review'):
            continue
        if 'start' not in m or 'end' not in m:
            return None
        if not m.get('category'):
            return None  # uncategorized: wait for the Phase 2 backfill
        kept.append(m)
    return kept


def fetch_episodes(conn):
    return conn.execute("""
        SELECT p.slug,
               COALESCE(p.title_override, p.title, p.slug) AS podcast_name,
               p.description AS podcast_description,
               e.episode_id, e.title AS episode_title,
               e.description AS episode_description,
               e.pending_review_count, e.processed_at,
               ed.ad_markers_json, ed.original_segments_json
        FROM episodes e
        JOIN podcasts p ON p.id = e.podcast_id
        JOIN episode_details ed ON ed.episode_id = e.id
        WHERE e.status = 'processed'
          AND ed.original_segments_json IS NOT NULL
          AND ed.original_segments_json NOT IN ('', 'null', '[]')
        ORDER BY p.slug, e.processed_at DESC
    """).fetchall()


def round_robin(rows, limit):
    by_feed = defaultdict(list)
    for row in rows:
        by_feed[row['slug']].append(row)
    picked, feeds = [], sorted(by_feed)
    i = 0
    while len(picked) < limit and any(by_feed.values()):
        feed = feeds[i % len(feeds)]
        if by_feed[feed]:
            picked.append(by_feed[feed].pop(0))
        i += 1
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', required=True, help='MinusPod SQLite DB copy')
    ap.add_argument('--minuspod-src', help='path to MinusPod src/')
    ap.add_argument('--limit', type=int, default=25,
                    help='max episodes (round-robin across feeds)')
    ap.add_argument('--license', default='unknown')
    ap.add_argument('--instance', default='primary')
    ap.add_argument('--teacher-model', default=None)
    ap.add_argument('--system-prompt-file',
                    help='use this system prompt instead of the static one')
    args = ap.parse_args()

    setup_minuspod_path(args.minuspod_src)
    from ad_detector import create_windows, format_window_prompt
    from ad_detector.prompts import get_static_system_prompt

    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding='utf-8')
    else:
        system_prompt = get_static_system_prompt()
    system_ref = store_prompt(system_prompt)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    win_size, win_overlap = read_window_settings(conn)
    holdout = load_holdout()
    excluded_feeds = load_excluded_feeds()
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    rows = fetch_episodes(conn)
    skipped = defaultdict(int)
    eligible = []
    for row in rows:
        if row['slug'] in excluded_feeds:
            skipped['excluded_feed'] += 1
            continue
        if (row['slug'], row['episode_id']) in holdout:
            skipped['holdout'] += 1
            continue
        if row['pending_review_count']:
            skipped['pending_review'] += 1
            continue
        eligible.append(row)

    stats = {'episodes': 0, 'windows': 0, 'empty_windows': 0, 'ads': 0}
    feeds_seen = set()
    for row in round_robin(eligible, args.limit):
        markers = usable_markers(json.loads(row['ad_markers_json'] or '[]'))
        if markers is None:
            skipped['unusable_markers'] += 1
            continue
        segments = json.loads(row['original_segments_json'])
        windows = create_windows(segments, window_size=win_size, overlap=win_overlap)
        if not windows:
            skipped['no_windows'] += 1
            continue

        description_section = build_description_section(
            row['podcast_description'], row['episode_description'])
        out_path = EXAMPLES_DIR / row['slug'] / f"{row['episode_id']}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open('w', encoding='utf-8') as f:
            for idx, window in enumerate(windows):
                lines = [f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}"
                         for seg in window['segments']]
                user_prompt = format_window_prompt(
                    podcast_name=row['podcast_name'],
                    episode_title=row['episode_title'] or '',
                    description_section=description_section,
                    transcript_lines=lines,
                    window_index=idx,
                    total_windows=len(windows),
                    window_start=window['start'],
                    window_end=window['end'],
                    audio_context='',
                )
                completion = window_completion(markers, window, window['segments'])
                example = {
                    'id': f"{row['slug']}/{row['episode_id']}/w{idx}",
                    'source': {
                        'feed': row['slug'],
                        'episode_id': row['episode_id'],
                        'window': idx,
                        'total_windows': len(windows),
                        'license': args.license,
                        'instance': args.instance,
                    },
                    'tier': 'machine_accepted',
                    'prompt': {
                        'system': system_ref,
                        'user': user_prompt,
                        'window_start': window['start'],
                        'window_end': window['end'],
                    },
                    'completion': completion,
                    'teacher': {'model': args.teacher_model},
                    'provenance': {
                        'reviewed': False,
                        'corrected': False,
                        'category_source': 'original',
                        'audio_context_omitted': True,
                        'extracted_at': now,
                        'extractor_version': EXTRACTOR_VERSION,
                    },
                }
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                stats['windows'] += 1
                stats['ads'] += len(completion)
                if not completion:
                    stats['empty_windows'] += 1
        stats['episodes'] += 1
        feeds_seen.add(row['slug'])

    print(f"episodes: {stats['episodes']} across {len(feeds_seen)} feeds")
    print(f"windows: {stats['windows']} ({stats['empty_windows']} empty, "
          f"{stats['empty_windows'] / max(stats['windows'], 1):.0%})")
    print(f"ad spans: {stats['ads']}")
    print(f"window config: {win_size:.0f}s size / {win_overlap:.0f}s overlap")
    print(f"system prompt: {system_ref}")
    for reason, count in sorted(skipped.items()):
        print(f"skipped ({reason}): {count}")
    overlap = feeds_seen & {slug for slug, _ in holdout}
    if overlap:
        print(f"WARNING: {len(overlap)} extracted feed(s) also appear in the "
              f"benchmark corpus (episode-level holdout enforced, but "
              f"benchmark scores on these shows partly measure in-domain "
              f"generalization): {', '.join(sorted(overlap))}")


if __name__ == '__main__':
    main()
