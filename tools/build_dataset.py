"""Build train/val chat-format JSONL from extracted examples.

Val split is by feed (unseen shows), not by random window.

Usage:
    uv run python tools/build_dataset.py --val-feeds feed-a,feed-b \
        [--out-dir .local] [--tier-weights machine_accepted=1.0]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, iter_examples, load_prompt  # noqa: E402


def parse_tier_weights(raw):
    weights = {}
    for part in filter(None, (raw or '').split(',')):
        tier, _, w = part.partition('=')
        weights[tier.strip()] = float(w)
    return weights


def to_chat(example, system_cache):
    ref = example['prompt']['system']
    if ref not in system_cache:
        system_cache[ref] = load_prompt(ref)
    # Compact array, exactly what the production parser expects back.
    assistant = json.dumps(example['completion'], ensure_ascii=False,
                           separators=(',', ':'))
    return {
        'messages': [
            {'role': 'system', 'content': system_cache[ref]},
            {'role': 'user', 'content': example['prompt']['user']},
            {'role': 'assistant', 'content': assistant},
        ]
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--val-feeds', required=True,
                    help='comma-separated feed slugs held out for validation')
    ap.add_argument('--out-dir', default=str(REPO_ROOT / '.local'))
    ap.add_argument('--tier-weights', default='',
                    help='tier=weight pairs; weight <1 downsamples, default 1.0')
    ap.add_argument('--seed', type=int, default=13)
    args = ap.parse_args()

    val_feeds = {f.strip() for f in args.val_feeds.split(',') if f.strip()}
    weights = parse_tier_weights(args.tier_weights)
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system_cache = {}
    counts = {'train': 0, 'val': 0, 'downsampled': 0}
    seen_feeds = set()
    with (out_dir / 'train.jsonl').open('w', encoding='utf-8') as train_f, \
         (out_dir / 'val.jsonl').open('w', encoding='utf-8') as val_f:
        for _, _, ex in iter_examples():
            feed = ex['source']['feed']
            seen_feeds.add(feed)
            if rng.random() >= weights.get(ex['tier'], 1.0):
                counts['downsampled'] += 1
                continue
            line = json.dumps(to_chat(ex, system_cache), ensure_ascii=False)
            if feed in val_feeds:
                val_f.write(line + '\n')
                counts['val'] += 1
            else:
                train_f.write(line + '\n')
                counts['train'] += 1

    missing = val_feeds - seen_feeds
    if missing:
        raise SystemExit(f"val feeds not found in examples: {sorted(missing)}")
    print(f"train: {counts['train']}  val: {counts['val']}  "
          f"downsampled: {counts['downsampled']}")
    print(f"wrote {out_dir / 'train.jsonl'} and {out_dir / 'val.jsonl'}")


if __name__ == '__main__':
    main()
