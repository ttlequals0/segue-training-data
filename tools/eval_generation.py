"""Generation eval on the feed-held-out val set.

Scores JSON compliance, per-break span P/R/F0.5 at IoU >= 0.5 (mirrors the
MinusPod benchmark scorer), no-ad false positives, and boundary MAE.

Usage:
    uv run python tools/eval_generation.py --run-id r1 --revision <sha> \
        --adapter .local/runs/r1/adapter
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render  # noqa: E402
from common import REPO_ROOT  # noqa: E402
from spans import match_spans, merge_gaps  # noqa: E402


def parse_prediction(text):
    """Strict parse: a bare JSON array of span objects, else None."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    for ad in data:
        if not isinstance(ad, dict) or not {'start', 'end'} <= set(ad):
            return None
        if not all(isinstance(ad[k], (int, float)) and not isinstance(ad[k], bool)
                   for k in ('start', 'end')):
            return None
    return data


TIER_BANDS = REPO_ROOT / 'data' / 'tier_bands.json'


def tier_band(f05, floors):
    """Which published tier band this F0.5 lands in.

    The benchmark assigns tiers by a paired test against each tier's leader,
    so this is where a score falls against the published roster, not a tier.
    """
    for letter, floor in sorted(floors.items(), key=lambda kv: -kv[1]):
        if f05 >= floor:
            return letter
    return sorted(floors, key=lambda k: floors[k])[0]


def _ranges(ads):
    normed = [{'start': a['start'], 'end': a['end'],
               'confidence': float(a.get('confidence', 0.0))} for a in ads]
    return [(s['start'], s['end']) for s in merge_gaps(normed)]


def score(rows, predictions):
    tp = fp = fn = parsed = 0
    noad_windows = noad_fp = 0
    start_deltas, end_deltas = [], []
    for row, pred in zip(rows, predictions):
        truth = _ranges(json.loads(row['messages'][2]['content']))
        if not truth:
            noad_windows += 1
        ads = parse_prediction(pred)
        if ads is None:
            fn += len(truth)
            continue
        parsed += 1
        spans = _ranges(ads)
        if not truth:
            noad_fp += len(spans)
            fp += len(spans)
            continue
        matches = match_spans(spans, truth)
        tp += len(matches)
        fp += len(spans) - len(matches)
        fn += len(truth) - len(matches)
        for pi, ti, _ in matches:
            start_deltas.append(abs(spans[pi][0] - truth[ti][0]))
            end_deltas.append(abs(spans[pi][1] - truth[ti][1]))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denom = 0.25 * precision + recall
    return {
        'n': len(rows),
        'json_compliance': round(parsed / len(rows), 4) if rows else 0.0,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f05': round(1.25 * precision * recall / denom, 4) if denom else 0.0,
        'tp': tp, 'fp': fp, 'fn': fn,
        'noad_windows': noad_windows, 'noad_fp': noad_fp,
        'start_mae': round(statistics.fmean(start_deltas), 2)
                     if start_deltas else None,
        'end_mae': round(statistics.fmean(end_deltas), 2)
                   if end_deltas else None,
    }


def generate_all(model, tokenizer, rows, model_name, max_new_tokens=1024):
    import torch
    outputs = []
    for row in rows:
        prefix, _ = render.render_texts(tokenizer, row['messages'], model_name)
        ids = tokenizer(prefix, add_special_tokens=False,
                        return_tensors='pt').to(model.device)
        # Pass the mask explicitly: without it transformers warns on every
        # call and falls back to inferring one, which it cannot do when pad
        # and eos are the same token.
        ids.setdefault('attention_mask', torch.ones_like(ids['input_ids']))
        with torch.no_grad():
            out = model.generate(**ids, do_sample=False,
                                 max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id
                                 if tokenizer.pad_token_id is not None
                                 else tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids['input_ids'].shape[1]:],
                                skip_special_tokens=True)
        outputs.append(text.strip())
    return outputs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--val', default=str(REPO_ROOT / '.local' / 'val.jsonl'))
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--revision', required=True)
    ap.add_argument('--adapter', help='PEFT adapter dir (omit with --merged)')
    ap.add_argument('--merged', help='merged model dir (omit with --adapter)')
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    if bool(args.adapter) == bool(args.merged):
        raise SystemExit('pass exactly one of --adapter or --merged')

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(args.val, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.merged:
        tokenizer = AutoTokenizer.from_pretrained(args.merged)
        model = AutoModelForCausalLM.from_pretrained(
            args.merged, dtype=torch.bfloat16, device_map='auto')
    else:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.model,
                                                  revision=args.revision)
        base = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16,
            device_map='auto')
        model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    predictions = generate_all(model, tokenizer, rows, args.model)
    result = score(rows, predictions)
    if TIER_BANDS.exists():
        bands = json.loads(TIER_BANDS.read_text())
        result['tier_band'] = tier_band(result['f05'], bands['floors'])
        result['tier_band_source'] = bands['minuspod_version']
    out = REPO_ROOT / '.local' / f'eval-{args.run_id}.json'
    out.write_text(json.dumps(
        {'args': vars(args),
         'metrics': result}, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if 'tier_band' in result:
        print(f"\nBand {result['tier_band']} against the "
              f"{result['tier_band_source']} published roster. This is a "
              f"held-out split, not the benchmark corpus, so it indicates "
              f"where the score lands rather than earning a tier. Run the "
              f"benchmark harness for a comparable row.")
    print(f'written to {out}')


if __name__ == '__main__':
    main()
