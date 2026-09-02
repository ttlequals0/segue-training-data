"""Export the trained adapter and a merged BF16 model, then verify equivalence.

The gate: greedy generations from base+adapter and from the merged model
must be token-identical on fixture windows before the export is accepted.

Usage:
    uv run python tools/export_local.py --run-id r1 --revision <sha> \
        --adapter .local/runs/r1/adapter
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as manifest_mod  # noqa: E402
import render  # noqa: E402
from common import REPO_ROOT, sha256_file  # noqa: E402


def checksum_dir(path):
    path = Path(path)
    return {str(p.relative_to(path)): sha256_file(p)
            for p in sorted(path.rglob('*')) if p.is_file()}


def generations_match(model_a, model_b, input_ids, max_new_tokens=128):
    import torch
    outs = []
    for model in (model_a, model_b):
        ids = input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids,
                                 attention_mask=torch.ones_like(ids),
                                 do_sample=False,
                                 max_new_tokens=max_new_tokens)
        outs.append(out[0].tolist())
    return outs[0] == outs[1]


def compute_logit_max_diff(model_a, model_b, ids_list):
    import torch
    max_diff = 0.0
    for ids in ids_list:
        with torch.no_grad():
            logits_a = model_a(ids.to(model_a.device)).logits
            logits_b = model_b(ids.to(model_b.device)).logits
        diff = float((logits_a.cpu() - logits_b.cpu()).abs().max())
        max_diff = max(max_diff, diff)
    return max_diff


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--revision', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--fixtures', default=str(REPO_ROOT / '.local' / 'val.jsonl'))
    ap.add_argument('--n-fixtures', type=int, default=8)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = Path(args.out or REPO_ROOT / '.local' / 'export' / args.run_id)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              revision=args.revision)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        device_map='auto')
    adapter_model = PeftModel.from_pretrained(base, args.adapter).eval()

    out.mkdir(parents=True, exist_ok=True)
    adapter_out = out / 'adapter'
    if adapter_out.exists():
        shutil.rmtree(adapter_out)
    shutil.copytree(args.adapter, adapter_out)

    with open(args.fixtures, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    fixtures = rows[:args.n_fixtures]
    fixture_ids = []
    for row in fixtures:
        prefix, _ = render.render_texts(tokenizer, row['messages'], args.model)
        fixture_ids.append(tokenizer(prefix, add_special_tokens=False,
                                     return_tensors='pt')['input_ids'])

    if not fixture_ids:
        raise SystemExit(f'no fixtures loaded from {args.fixtures}')

    merged = adapter_model.merge_and_unload()
    merged_out = out / 'merged'
    merged.save_pretrained(str(merged_out), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_out))
    del adapter_model, merged, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reloaded = AutoModelForCausalLM.from_pretrained(
        str(merged_out), dtype=torch.bfloat16, device_map='auto').eval()
    base2 = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        device_map='auto')
    adapter2 = PeftModel.from_pretrained(base2, str(adapter_out)).eval()

    results = [generations_match(adapter2, reloaded, ids)
               for ids in fixture_ids]
    logit_max_diff = compute_logit_max_diff(adapter2, reloaded, fixture_ids)

    export_manifest = manifest_mod.build_manifest(
        vars(args), {},
        {'adapter_sha256': checksum_dir(adapter_out),
         'merged_sha256': checksum_dir(merged_out),
         'fixture_matches': results,
         'logit_max_diff': logit_max_diff})
    (out / 'export_manifest.json').write_text(
        json.dumps(export_manifest, indent=2) + '\n')
    print(f'fixtures matched: {sum(results)}/{len(results)}, '
          f'logit max diff {logit_max_diff:.2e}')
    if not all(results):
        raise SystemExit('EQUIVALENCE FAILED: merged model diverges from '
                         'base+adapter; do not publish this export')
    print(f'export verified; artifacts in {out}')


if __name__ == '__main__':
    main()
