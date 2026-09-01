"""One-shot label migration to the per-break output contract.

Drops outro spans (no phase 1 prompt has SHOW SEGMENTS), drops
audio-only-evidence spans unless kept by data/keep_spans.json, merges
sub-15s gaps, and recomputes every end_text from the window transcript.

Usage:
    uv run python tools/fix_labels.py --dry-run   # review drops first
    uv run python tools/fix_labels.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXAMPLES_DIR, REPO_ROOT  # noqa: E402
from spans import GAP_SECONDS, merge_gaps  # noqa: E402

KEEP_FILE = REPO_ROOT / 'data' / 'keep_spans.json'
ALLOWED_CATEGORIES = {'sponsor', 'cross_promo', 'self_promo', 'interaction'}
AUDIO_ONLY = re.compile(r'no other ad signal', re.IGNORECASE)
SEGMENT_LINE = re.compile(r'^\[(\d+(?:\.\d+)?)s - (\d+(?:\.\d+)?)s\] (.*)$')


def parse_segments(user_prompt):
    segs = []
    for line in user_prompt.splitlines():
        m = SEGMENT_LINE.match(line.strip())
        if m:
            segs.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    return segs


def recompute_end_text(segments, span_start, span_end, n=5):
    """Last `n` transcript words from segments ending inside the span."""
    words = []
    for s, e, text in segments:
        if e <= span_start or s >= span_end:
            continue
        if e <= span_end + 2.0:
            words.extend(text.split())
    if not words:
        words = [w for s, e, text in segments
                 if s < span_end and e > span_start for w in text.split()]
    return ' '.join(words[-n:])


def load_keep_list():
    if not KEEP_FILE.exists():
        return {}
    entries = json.loads(KEEP_FILE.read_text(encoding='utf-8'))
    return {(e['id'], float(e['start'])): e['reason'] for e in entries}


def fix_example(ex, keep):
    """Returns (fixed example, dropped spans, fix tags applied)."""
    dropped, spans, fixes = [], [], []
    for ad in ex['completion']:
        if ad['category'] not in ALLOWED_CATEGORIES:
            dropped.append({**ad, 'rule': 'outro_without_show_segments'})
            continue
        if AUDIO_ONLY.search(ad['reason']):
            new_reason = keep.get((ex['id'], float(ad['start'])))
            if new_reason is None:
                dropped.append({**ad, 'rule': 'audio_only_evidence'})
                continue
            ad = {**ad, 'reason': new_reason}
            if 'reason_corrected' not in fixes:
                fixes.append('reason_corrected')
        spans.append(dict(ad))
    fixes = list(dict.fromkeys([d['rule'] for d in dropped] + fixes))
    n_before = len(spans)
    spans = merge_gaps(spans, GAP_SECONDS)
    if len(spans) != n_before:
        fixes.append('merged_gaps')
    segments = parse_segments(ex['prompt']['user'])
    for ad in spans:
        new_text = recompute_end_text(segments, ad['start'], ad['end'])
        if new_text and new_text != ad['end_text']:
            ad['end_text'] = new_text
            if 'end_text_recomputed' not in fixes:
                fixes.append('end_text_recomputed')
    out = dict(ex)
    out['completion'] = spans
    if dropped or fixes:
        prov = dict(out['provenance'])
        if dropped:
            prov['dropped_spans'] = [
                {'start': d['start'], 'end': d['end'], 'category': d['category'],
                 'reason': d['reason'], 'rule': d['rule']} for d in dropped]
        if fixes:
            prov['label_fixes'] = fixes
        out['provenance'] = prov
    return out, dropped, fixes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    keep = load_keep_list()
    stats = {'examples': 0, 'changed': 0, 'dropped': 0}
    for path in sorted(EXAMPLES_DIR.rglob('*.jsonl')):
        lines_out, file_changed = [], False
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            stats['examples'] += 1
            fixed, dropped, fixes = fix_example(ex, keep)
            for d in dropped:
                segs = parse_segments(ex['prompt']['user'])
                tail = recompute_end_text(segs, d['start'], d['end'])
                print(f"DROP {ex['id']} [{d['start']}-{d['end']}] "
                      f"{d['rule']} reason={d['reason']!r} "
                      f"transcript_tail={tail!r}")
            stats['dropped'] += len(dropped)
            if fixes or dropped:
                stats['changed'] += 1
                file_changed = True
            lines_out.append(json.dumps(fixed, ensure_ascii=False))
        if file_changed and not args.dry_run:
            tmp = path.with_suffix('.jsonl.tmp')
            tmp.write_text('\n'.join(lines_out) + '\n', encoding='utf-8')
            tmp.replace(path)
    mode = 'dry-run' if args.dry_run else 'applied'
    print(f"{mode}: {stats['examples']} examples, {stats['changed']} changed, "
          f"{stats['dropped']} spans dropped, {len(keep)} keep-list entries")


if __name__ == '__main__':
    main()
