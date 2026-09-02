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
from common import REPO_ROOT, iter_examples, load_prompt, sha256_file  # noqa: E402


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


def split_examples(examples, val_feeds, weights, rng):
    """Route first, then downsample train only; val is never sampled."""
    train, val, downsampled = [], [], 0
    for ex in examples:
        if ex['source']['feed'] in val_feeds:
            val.append(ex)
        elif rng.random() >= weights.get(ex['tier'], 1.0):
            downsampled += 1
        else:
            train.append(ex)
    return train, val, downsampled


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

    examples = [ex for _, _, ex in iter_examples()]
    seen_feeds = {ex['source']['feed'] for ex in examples}
    missing = val_feeds - seen_feeds
    if missing:
        raise SystemExit(f"val feeds not found in examples: {sorted(missing)}")

    train, val, downsampled = split_examples(examples, val_feeds, weights, rng)
    system_cache = {}
    splits = {'train': train, 'val': val}
    for name, rows in splits.items():
        tmp = out_dir / f'{name}.jsonl.tmp'
        with tmp.open('w', encoding='utf-8') as f:
            for ex in rows:
                f.write(json.dumps(to_chat(ex, system_cache),
                                   ensure_ascii=False) + '\n')
        tmp.replace(out_dir / f'{name}.jsonl')

    manifest = {
        'val_feeds': sorted(val_feeds),
        'seed': args.seed,
        'tier_weights': weights,
        'counts': {name: len(rows) for name, rows in splits.items()},
        'downsampled': downsampled,
        'ids': {name: [ex['id'] for ex in rows] for name, rows in splits.items()},
        'sha256': {name: sha256_file(out_dir / f'{name}.jsonl') for name in splits},
    }
    tmp = out_dir / 'split_manifest.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    tmp.replace(out_dir / 'split_manifest.json')

    print(f"train: {len(train)}  val: {len(val)}  downsampled: {downsampled}")
    print(f"wrote train.jsonl, val.jsonl, split_manifest.json in {out_dir}")


if __name__ == '__main__':
    main()
