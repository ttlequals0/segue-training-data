"""Preflight gate: refuse to train until data, renderer, and device check out.

Writes .local/preflight.json; train_local.py requires a stamp whose hashes
match its inputs.

Usage:
    uv run python tools/preflight.py --model Qwen/Qwen3.5-9B --revision <sha>
    uv run python tools/preflight.py ... --device cpu --skip-model-step
"""
import argparse
import datetime
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render  # noqa: E402
from common import REPO_ROOT, sha256_file  # noqa: E402

STAMP = REPO_ROOT / '.local' / 'preflight.json'


def parse_memory_fraction(value):
    """Argparse type: a device-memory cap in (0, 1]."""
    f = float(value)
    if not 0.0 < f <= 1.0:
        raise argparse.ArgumentTypeError(
            f'memory fraction must be in (0, 1], got {value}')
    return f


def load_rows(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f'{path}: no examples')
    return rows


def check_disjoint(manifest_path, train_path, val_path):
    if not Path(manifest_path).exists():
        return f'{manifest_path} missing; rebuild the dataset'
    manifest = json.loads(Path(manifest_path).read_text())
    ids = manifest['ids']
    overlap = set(ids['train']) & set(ids['val'])
    if overlap:
        return f'train/val overlap: {sorted(overlap)[:5]}'

    # Verify manifest sha256 hashes match actual files
    if 'sha256' not in manifest:
        return f'{manifest_path} missing sha256; rebuild the dataset'
    manifest_sha256 = manifest['sha256']
    train_hash = sha256_file(train_path)
    val_hash = sha256_file(val_path)

    if manifest_sha256.get('train') != train_hash:
        return f'{manifest_path} is stale for train; rebuild the dataset'
    if manifest_sha256.get('val') != val_hash:
        return f'{manifest_path} is stale for val; rebuild the dataset'
    return None


def render_all(tokenizer, rows, model_name, max_length):
    lengths, categories, empty = [], {}, 0
    for row in rows:
        enc = render.encode_example(tokenizer, row['messages'],
                                    model_name, max_length)
        lengths.append(enc['length'])
        ads = json.loads(row['messages'][2]['content'])
        if not ads:
            empty += 1
        for ad in ads:
            categories[ad['category']] = categories.get(ad['category'], 0) + 1
    return {
        'n': len(rows),
        'tokens_min': min(lengths),
        'tokens_median': statistics.median(lengths),
        'tokens_max': max(lengths),
        'categories': categories,
        'empty_fraction': round(empty / len(rows), 3),
    }


def expected_steps(n, batch_size, grad_accum, epochs):
    return max(1, math.ceil(n / (batch_size * grad_accum))) * epochs


def check(name, fn, failures):
    try:
        result = fn()
        if isinstance(result, list):
            print(f'PASS {name}: {len(result)} rows')
        else:
            print(f'PASS {name}' + (f': {result}' if result else ''))
        return result
    except (Exception, SystemExit) as e:
        print(f'FAIL {name}: {e}')
        failures.append(name)
        return None


def optimizer_step_smoke(model_name, revision, attn, tokenizer, rows,
                         model_max_length, memory_fraction):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM
    # Unified memory is host RAM: an uncapped allocation reaches the kernel OOM
    # killer and takes down the box instead of raising a catchable CUDA error.
    torch.cuda.set_per_process_memory_fraction(memory_fraction)

    def mark(stage):
        print(f'  {stage}: peak '
              f'{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB',
              flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=torch.bfloat16,
        attn_implementation=attn, device_map='cuda')
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, bias='none', target_modules='all-linear'))
    # Mirror the trainer's memory configuration, or the smoke step measures a
    # setup we never run: without checkpointing a 16k-token window holds every
    # layer's activations at once.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    # HF gates checkpointing on `self.gradient_checkpointing and self.training`,
    # and from_pretrained returns an eval-mode model, so without train() the
    # checkpointing above is silently inert.
    model.train()
    active = any(getattr(m, 'gradient_checkpointing', False)
                 for m in model.modules())
    print(f'  gradient checkpointing active: {active} '
          f'(training={model.training})', flush=True)
    if not active:
        raise RuntimeError('gradient checkpointing did not take effect; '
                           'the smoke step would not match training')
    longest = max(rows, key=lambda r: len(r['messages'][1]['content']))
    enc = render.encode_example(tokenizer, longest['messages'],
                                model_name, model_max_length)
    ids = torch.tensor([enc['input_ids']], device='cuda')
    labels = torch.tensor([enc['labels']], device='cuda')
    opt = torch.optim.AdamW((p for p in model.parameters()
                             if p.requires_grad), lr=1e-4)
    mark('model loaded')
    loss = model(input_ids=ids, labels=labels).loss
    mark('after forward')
    loss.backward()
    mark('after backward')
    opt.step()
    mark('after optimizer step')
    peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30
    return {'loss': float(loss), 'peak_memory_gb': round(peak_gb, 1),
            'memory_fraction': memory_fraction,
            'tokens': enc['length']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', default=str(REPO_ROOT / '.local' / 'train.jsonl'))
    ap.add_argument('--val', default=str(REPO_ROOT / '.local' / 'val.jsonl'))
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--revision', required=True)
    ap.add_argument('--max-length', type=int, default=16384)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--memory-fraction', type=parse_memory_fraction,
                    default=0.8,
                    help='cap on device memory; unified memory is host '
                         'RAM, so an uncapped run can OOM-kill the box')
    ap.add_argument('--skip-model-step', action='store_true')
    args = ap.parse_args()

    failures = []

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              revision=args.revision)
    train_rows = check('train file', lambda: load_rows(args.train), failures) or []
    val_rows = check('val file', lambda: load_rows(args.val), failures) or []
    manifest_path = Path(args.train).parent / 'split_manifest.json'

    def disjoint():
        err = check_disjoint(manifest_path, args.train, args.val)
        if err:
            raise SystemExit(err)

    check('split disjoint', disjoint, failures)
    train_stats = check('render train', lambda: render_all(
        tokenizer, train_rows, args.model, args.max_length), failures)
    val_stats = check('render val', lambda: render_all(
        tokenizer, val_rows, args.model, args.max_length), failures)
    steps = expected_steps(len(train_rows), args.batch_size,
                           args.grad_accum, args.epochs)
    print(f'INFO expected optimizer steps: {steps}')

    attn = 'sdpa'
    if args.device == 'cuda':
        import torch
        check('cuda visible', lambda: torch.cuda.get_device_name(0), failures)

        def bf16():
            if not torch.cuda.is_bf16_supported():
                raise SystemExit('bf16 unsupported')

        check('bf16 supported', bf16, failures)
        try:
            import flash_attn  # noqa: F401
            attn = 'flash_attention_2'
        except ImportError:
            pass
        print(f'INFO attention implementation: {attn}')
        if not args.skip_model_step:
            check('optimizer step', lambda: optimizer_step_smoke(
                args.model, args.revision, attn, tokenizer, train_rows,
                args.max_length, args.memory_fraction), failures)

    if failures:
        raise SystemExit(f'preflight FAILED: {failures}')

    stamp = {
        'train_sha256': sha256_file(args.train),
        'val_sha256': sha256_file(args.val),
        'model': args.model,
        'revision': args.revision,
        'attn': attn,
        'max_length': args.max_length,
        'template_hash': render.template_hash(tokenizer),
        'passed_at': datetime.datetime.now(datetime.timezone.utc)
                     .strftime('%Y-%m-%dT%H:%M:%SZ'),
        'stats': {'train': train_stats, 'val': val_stats,
                  'expected_steps': steps},
    }
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(json.dumps(stamp, indent=2) + '\n')
    print(f'preflight PASSED; stamp written to {STAMP}')


if __name__ == '__main__':
    main()
