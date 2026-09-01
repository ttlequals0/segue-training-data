"""LoRA-train Segue locally (DGX Spark) with Transformers + PEFT.

Requires a passing preflight stamp (tools/preflight.py) for the same data,
model, and revision.

Usage:
    uv run python tools/train_local.py --run-id r1 --revision <commit-sha>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as manifest_mod  # noqa: E402
import render  # noqa: E402
from common import REPO_ROOT, sha256_file  # noqa: E402

STAMP = REPO_ROOT / '.local' / 'preflight.json'
DEFAULT_MODEL = 'Qwen/Qwen3.5-9B'


def build_lora_config(r, alpha, dropout):
    from peft import LoraConfig, TaskType
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha,
                      lora_dropout=dropout, bias='none',
                      target_modules='all-linear')


def expected_lora_params(model, r):
    """Params all-linear LoRA adds: r * (in + out) per Linear, minus lm_head."""
    import torch.nn as nn
    total = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and 'lm_head' not in name:
            total += r * (mod.in_features + mod.out_features)
    return total


def make_collator(pad_id):
    def collate(features):
        import torch
        width = max(f['length'] for f in features)
        ids, labels, mask = [], [], []
        for f in features:
            pad = width - f['length']
            ids.append(f['input_ids'] + [pad_id] * pad)
            labels.append(f['labels'] + [-100] * pad)
            mask.append([1] * f['length'] + [0] * pad)
        return {'input_ids': torch.tensor(ids),
                'labels': torch.tensor(labels),
                'attention_mask': torch.tensor(mask)}
    return collate


def require_stamp(train_path, val_path, model, revision):
    if not STAMP.exists():
        raise SystemExit('no preflight stamp; run tools/preflight.py first')
    stamp = json.loads(STAMP.read_text())
    expect = {'train_sha256': sha256_file(train_path),
              'val_sha256': sha256_file(val_path),
              'model': model, 'revision': revision}
    for key, want in expect.items():
        if stamp.get(key) != want:
            raise SystemExit(
                f'preflight stamp mismatch on {key}; rerun tools/preflight.py')
    return stamp


def load_rows(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', default=str(REPO_ROOT / '.local' / 'train.jsonl'))
    ap.add_argument('--val', default=str(REPO_ROOT / '.local' / 'val.jsonl'))
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--resume', action='store_true',
                    help='resume from the latest checkpoint in the run dir')
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=0.05)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--max-length', type=int, default=16384)
    ap.add_argument('--seed', type=int, default=13)
    args = ap.parse_args()

    stamp = require_stamp(args.train, args.val, args.model, args.revision)
    run_dir = REPO_ROOT / '.local' / 'runs' / args.run_id
    if run_dir.exists() and not args.resume:
        raise SystemExit(f'{run_dir} exists; pass --resume or a new --run-id')
    if args.resume and not run_dir.exists():
        raise SystemExit(f'--resume but {run_dir} does not exist')
    run_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              revision=args.revision)
    if render.template_hash(tokenizer) != stamp['template_hash']:
        raise SystemExit('chat template changed since preflight; rerun it')
    train_enc = [render.encode_example(tokenizer, r['messages'], args.model,
                                       args.max_length)
                 for r in load_rows(args.train)]
    val_enc = [render.encode_example(tokenizer, r['messages'], args.model,
                                     args.max_length)
               for r in load_rows(args.val)]

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        attn_implementation=stamp['attn'])
    expected = expected_lora_params(model, args.rank)
    model = get_peft_model(model, build_lora_config(
        args.rank, args.alpha, args.dropout))
    model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    matched = sorted({n.split('.lora_')[0].rsplit('.', 1)[-1]
                      for n, _ in model.named_parameters() if 'lora_' in n})
    print(f'LoRA modules: {matched}')
    print(f'trainable params: {trainable:,} (expected {expected:,})')
    if trainable != expected:
        raise SystemExit('trainable parameter count does not match all-linear '
                         'expectation; inspect target_modules')

    run_manifest = manifest_mod.build_manifest(
        vars(args),
        {'train': args.train, 'val': args.val,
         'lockfile': REPO_ROOT / 'uv.lock'},
        {'status': 'running', 'env': manifest_mod.collect_env(),
         'template_hash': stamp['template_hash'], 'attn': stamp['attn'],
         'lora_modules': matched, 'trainable_params': trainable,
         'run_dir': str(run_dir)})
    manifest_path = REPO_ROOT / 'runs' / f'{args.run_id}.json'
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + '\n')

    targs = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type='linear',
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        logging_steps=1,
        eval_strategy='steps',
        eval_steps=10,
        save_steps=20,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_enc,
                      eval_dataset=val_enc,
                      data_collator=make_collator(
                          tokenizer.pad_token_id or tokenizer.eos_token_id))
    result = trainer.train(resume_from_checkpoint=args.resume)
    final_eval = trainer.evaluate()

    adapter_dir = run_dir / 'adapter'
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    run_manifest.update({
        'status': 'completed',
        'train_loss': result.training_loss,
        'final_eval': final_eval,
        'best_checkpoint': trainer.state.best_model_checkpoint,
        'adapter_sha256': {p.name: sha256_file(p)
                           for p in sorted(adapter_dir.iterdir())
                           if p.is_file()},
    })
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + '\n')
    print(f'done; adapter in {adapter_dir}, manifest in {manifest_path}')


if __name__ == '__main__':
    main()
