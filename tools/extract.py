"""Extract training examples from a MinusPod database copy.

Reconstructs the per-window detection prompts with MinusPod's own
create_windows/format_window_prompt (the DB stores only a placeholder for
first_pass_prompt) and pairs each window with the final ad markers that
intersect it.

Usage:
    uv run python tools/extract.py --db .local/minuspod.db [--limit 25]

--limit 0 takes every eligible episode.
"""
import argparse
import datetime
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    EXAMPLES_DIR, EXTRACTOR_VERSION, load_excluded_feeds, load_holdout,
    setup_minuspod_path, store_prompt,
)

WINDOW_SIZE_DEFAULT = 600.0
WINDOW_OVERLAP_DEFAULT = 180.0
MIN_AD_OVERLAP_SECONDS = 1.0
BOUNDS_TOLERANCE_SECONDS = 0.5
# Fraction of a marker a correction must cover to target it (MinusPod's
# CORRECTION_MATCH_MIN_COVERAGE rule).
MATCH_COVERAGE = 0.5
AUTO_APPROVED_PREFIX = 'auto-approved'  # written by MinusPod process_episode on pass-2 corroboration


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
        clipped = clip(m, window)
        if clipped is None:
            continue
        s, e = clipped
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


def clip(span, window):
    """(start, end) of span inside window, or None if the overlap is too short."""
    s = max(float(span['start']), window['start'])
    e = min(float(span['end']), window['end'])
    return (s, e) if e - s >= MIN_AD_OVERLAP_SECONDS else None


def partition_markers(markers):
    """Split into (cut, blocking, undecided); None when a marker has no bounds.

    Blocking markers are cut without a category. Undecided ones are uncut and
    pending review (MinusPod's is_pending_review rule, inlined), kept by feed
    policy, or confidence-gated REVIEW ads nobody ruled on; they block unless
    a correction targets them."""
    cut, blocking, undecided = [], [], []
    for m in markers:
        if 'start' not in m or 'end' not in m:
            return None
        if m.get('was_cut', True):
            (cut if m.get('category') else blocking).append(m)
            continue
        gated = ((m.get('validation') or {}).get('decision') == 'REVIEW'
                 and not m.get('reviewer_verdict'))
        if m.get('held_for_review') or m.get('action_applied') == 'keep' or gated:
            undecided.append(m)
    return cut, blocking, undecided


def parse_bounds(raw):
    try:
        b = json.loads(raw)
        return {'start': float(b['start']), 'end': float(b['end'])}
    except (TypeError, ValueError, KeyError):
        return None


def fetch_corrections(conn):
    """Corrections keyed by episode_id, oldest first; a re-filed decision
    takes its newest id. Auto-approvals are not human."""
    by_episode = defaultdict(list)
    rows = conn.execute("""
        SELECT episode_id, correction_type, original_bounds, corrected_bounds,
               text_snippet
        FROM pattern_corrections
        WHERE correction_type IN ('confirm', 'false_positive',
                                  'boundary_adjustment', 'create')
        GROUP BY episode_id, correction_type, original_bounds, corrected_bounds,
                 text_snippet
        ORDER BY MAX(id)
    """)
    for row in rows:
        by_episode[row['episode_id']].append({
            'type': row['correction_type'],
            'original': parse_bounds(row['original_bounds']),
            'corrected': parse_bounds(row['corrected_bounds']),
            'human': not (row['correction_type'] == 'confirm' and
                          (row['text_snippet'] or '').startswith(AUTO_APPROVED_PREFIX)),
        })
    return by_episode


def overlap(a, b):
    return max(0.0, min(float(a['end']), float(b['end']))
               - max(float(a['start']), float(b['start'])))


def same_span(bounds, marker):
    return (abs(bounds['start'] - float(marker['start'])) <= BOUNDS_TOLERANCE_SECONDS
            and abs(bounds['end'] - float(marker['end'])) <= BOUNDS_TOLERANCE_SECONDS)


def covers(bounds, marker):
    length = float(marker['end']) - float(marker['start'])
    return length > 0 and overlap(bounds, marker) / length >= MATCH_COVERAGE


def targeted(bounds, markers, relation):
    return [m for m in markers if relation(bounds, m)] if bounds else []


def resolve_corrections(corrections, markers):
    """(hits, stale): each hit is a marker with a keep, drop or block action.

    Newest decision on a marker wins. Auto-approvals only ever keep. After
    all decisions, a rejection that still stands blocks any cut marker it
    clips that no person ruled on."""
    hits, stale, rejections = {}, 0, []
    for c in corrections:
        label = c['type']
        if c['type'] == 'confirm' and c['corrected']:
            label = 'confirm_trimmed'
        if not c['human']:
            label = 'auto_' + label
        if c['type'] == 'false_positive':
            bounds = c['original']
            targets = []
            for m in (markers if bounds else []):
                was_cut = m.get('was_cut', True)
                if not was_cut and same_span(bounds, m):
                    targets.append((m, 'keep'))
                elif was_cut and covers(bounds, m):
                    targets.append((m, 'drop'))
            if bounds:
                rejections.append((bounds, label, [m for m, _ in targets]))
        else:
            target = c['corrected'] or c['original']
            exact = targeted(target, markers, same_span)
            if any(m.get('was_cut', True) for m in exact):
                targets = [(m, 'keep') for m in exact]
            elif not c['human']:
                continue
            else:
                targets = [(m, 'block') for m in exact
                           or targeted(target, markers, covers)
                           or targeted(c['original'], markers, covers)]
        if not targets:
            stale += 1
            continue
        for m, action in targets:
            hits[id(m)] = {'marker': m, 'label': label, 'action': action}
    for bounds, label, targets in rejections:
        if any(hits[id(m)]['label'] != label for m in targets):
            continue
        blocked = False
        for m in markers:
            ruled = hits.get(id(m))
            if ruled and not ruled['label'].startswith('auto_'):
                continue
            if m.get('was_cut', True) and clip(m, bounds):
                hits[id(m)] = {'marker': m, 'label': label, 'action': 'block'}
                blocked = True
        if blocked and not targets:
            stale -= 1
    return list(hits.values()), stale


def classify_window(labels):
    """(tier, reviewed, corrected) from the correction labels touching a window."""
    human = {label for label in labels if not label.startswith('auto_')}
    if human - {'false_positive'}:
        tier = 'human_verified'
    elif human:
        tier = 'hard_negative'
    else:
        tier = 'machine_accepted'
    return tier, bool(human), bool(human - {'confirm'})


def fetch_episodes(conn):
    return conn.execute("""
        SELECT e.id, p.slug,
               COALESCE(p.title_override, p.title, p.slug) AS podcast_name,
               p.description AS podcast_description,
               e.episode_id, e.title AS episode_title,
               e.description AS episode_description,
               e.processed_at
        FROM episodes e
        JOIN podcasts p ON p.id = e.podcast_id
        JOIN episode_details ed ON ed.episode_id = e.id
        WHERE e.status = 'processed'
          AND e.pending_recut_at IS NULL
          AND ed.original_segments_json IS NOT NULL
          AND ed.original_segments_json NOT IN ('', 'null', '[]')
        ORDER BY p.slug, e.processed_at DESC
    """).fetchall()


def fetch_details(conn, row_id):
    """(markers, segments) for one episode; the blobs are too big to hold for all."""
    row = conn.execute(
        "SELECT ad_markers_json, original_segments_json FROM episode_details "
        "WHERE episode_id = ?", (row_id,)).fetchone()
    return (json.loads(row['ad_markers_json'] or '[]'),
            json.loads(row['original_segments_json']))


def format_counts(counter):
    return ", ".join(f"{k}={n}" for k, n in sorted(counter.items()))


def round_robin(rows):
    by_feed = defaultdict(list)
    for row in rows:
        by_feed[row['slug']].append(row)
    feeds = sorted(by_feed)
    i = 0
    while any(by_feed.values()):
        feed = feeds[i % len(feeds)]
        if by_feed[feed]:
            yield by_feed[feed].pop(0)
        i += 1


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
    skipped = Counter()
    eligible = []
    for row in rows:
        if row['slug'] in excluded_feeds:
            skipped['excluded_feed'] += 1
            continue
        if (row['slug'], row['episode_id']) in holdout:
            skipped['holdout'] += 1
            continue
        eligible.append(row)

    corrections = fetch_corrections(conn)
    stats = Counter()
    # pattern_corrections has no feed column, so a shared episode_id is ambiguous.
    ambiguous = conn.execute("""
        SELECT episode_id FROM episodes
        GROUP BY episode_id HAVING COUNT(DISTINCT podcast_id) > 1
    """).fetchall()
    for (episode_id,) in ambiguous:
        stats['ambiguous_id_corrections'] += len(corrections.pop(episode_id, []))
    tiers = Counter()
    labels = Counter()
    feeds_seen = set()
    for row in round_robin(eligible):
        if args.limit and stats['episodes'] >= args.limit:
            break
        all_markers, segments = fetch_details(conn, row['id'])
        parts = partition_markers(all_markers)
        if parts is None:
            skipped['unusable_markers'] += 1
            continue
        cut, blocking, undecided = parts
        hits, stale = resolve_corrections(
            corrections.get(row['episode_id'], []), all_markers)
        stats['stale_corrections'] += stale
        action = {id(h['marker']): h['action'] for h in hits}
        cut = [m for m in cut if action.get(id(m)) != 'drop']
        blocking = [m for m in blocking if action.get(id(m)) != 'drop']
        blocking += [m for m in undecided if id(m) not in action]
        blocking += [h['marker'] for h in hits if h['action'] == 'block']
        hits = [h for h in hits if h['action'] != 'block']
        windows = create_windows(segments, window_size=win_size, overlap=win_overlap)
        if not windows:
            skipped['no_windows'] += 1
            continue
        kept = [(idx, w) for idx, w in enumerate(windows)
                if not any(clip(b, w) for b in blocking)]
        skipped['blocked_windows'] += len(windows) - len(kept)
        if not kept:
            skipped['all_windows_blocked'] += 1
            continue

        description_section = build_description_section(
            row['podcast_description'], row['episode_description'])
        out_path = EXAMPLES_DIR / row['slug'] / f"{row['episode_id']}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open('w', encoding='utf-8') as f:
            for idx, window in kept:
                window_hits = [h for h in hits if clip(h['marker'], window)]
                window_labels = sorted({h['label'] for h in window_hits})
                tier, reviewed, corrected = classify_window(window_labels)
                provenance = {
                    'reviewed': reviewed,
                    'corrected': corrected,
                    'category_source': 'original',
                    'audio_context_omitted': True,
                    'extracted_at': now,
                    'extractor_version': EXTRACTOR_VERSION,
                }
                if window_labels:
                    provenance['corrections'] = window_labels
                dropped_spans = [
                    {'start': float(h['marker']['start']),
                     'end': float(h['marker']['end']),
                     'category': h['marker'].get('category') or 'uncategorized',
                     'reason': str(h['marker'].get('reason', '')),
                     'rule': 'rejected_but_cut'}
                    for h in window_hits if h['action'] == 'drop']
                if dropped_spans:
                    provenance['dropped_spans'] = dropped_spans
                for label in window_labels:
                    labels[label] += 1
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
                completion = window_completion(cut, window, window['segments'])
                tiers[tier] += 1
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
                    'tier': tier,
                    'prompt': {
                        'system': system_ref,
                        'user': user_prompt,
                        'window_start': window['start'],
                        'window_end': window['end'],
                    },
                    'completion': completion,
                    'teacher': {'model': args.teacher_model},
                    'provenance': provenance,
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
    print(f"tiers: {format_counts(tiers)}")
    print(f"corrections on windows: {format_counts(labels)}")
    print(f"stale corrections (no matching marker): {stats['stale_corrections']}")
    print(f"corrections on ambiguous episode ids: {stats['ambiguous_id_corrections']}")
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
