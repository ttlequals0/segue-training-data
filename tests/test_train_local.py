import json

import pytest

torch = pytest.importorskip("torch")

from train_local import (  # noqa: E402
    build_lora_config, expected_lora_params, make_collator, require_stamp,
    build_training_args,
)


def test_build_lora_config_fields():
    cfg = build_lora_config(16, 32, 0.05)
    assert cfg.r == 16 and cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05 and cfg.bias == "none"
    assert cfg.target_modules == "all-linear"
    assert str(cfg.task_type) in ("TaskType.CAUSAL_LM", "CAUSAL_LM")
    assert not cfg.modules_to_save


def test_expected_lora_params_counts_linear_not_lm_head():
    model = torch.nn.ModuleDict({
        "proj": torch.nn.Linear(8, 4),
        "lm_head": torch.nn.Linear(8, 100),
    })
    assert expected_lora_params(model, r=2) == 2 * (8 + 4)


def test_collator_pads_and_masks():
    collate = make_collator(pad_id=0)
    batch = collate([
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "length": 3},
        {"input_ids": [4], "labels": [4], "length": 1},
    ])
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_require_stamp_rejects_mismatch(tmp_path, monkeypatch):
    import train_local
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text("{}\n")
    val.write_text("{}\n")
    stamp = tmp_path / "preflight.json"
    monkeypatch.setattr(train_local, "STAMP", stamp)
    with pytest.raises(SystemExit, match="preflight"):
        require_stamp(train, val, "m", "rev", 16384)
    from common import sha256_file
    stamp.write_text(json.dumps({
        "train_sha256": sha256_file(train), "val_sha256": sha256_file(val),
        "model": "m", "revision": "rev", "attn": "sdpa",
        "max_length": 16384}))
    assert require_stamp(train, val, "m", "rev", 16384)["attn"] == "sdpa"
    train.write_text('{"changed": 1}\n')
    with pytest.raises(SystemExit, match="preflight"):
        require_stamp(train, val, "m", "rev", 16384)
    train.write_text("{}\n")
    with pytest.raises(SystemExit, match="preflight"):
        require_stamp(train, val, "m", "rev", 8192)


def _args(**over):
    base = {'epochs': 3, 'batch_size': 2, 'grad_accum': 8, 'lr': 1e-4,
            'seed': 13, 'eval_steps': 10, 'save_steps': 10}
    base.update(over)
    return type('Args', (), base)()


def test_validate_cadence_rejects_non_multiple():
    from train_local import validate_cadence
    # transformers rejects this only when TrainingArguments is constructed,
    # which needs bf16 hardware, so the check has to stand on its own.
    with pytest.raises(ValueError, match="multiple"):
        validate_cadence(10, 5)
    with pytest.raises(ValueError, match="positive"):
        validate_cadence(10, 0)


def test_validate_cadence_accepts_multiples():
    from train_local import validate_cadence
    for eval_steps, save_steps in ((10, 10), (10, 20), (5, 15), (1, 7)):
        validate_cadence(eval_steps, save_steps)


def test_build_training_args_eval_batch_matches_train(tmp_path):
    try:
        targs = build_training_args(tmp_path, _args(batch_size=1))
    except ValueError as e:
        pytest.skip(f"bf16 unsupported on this box: {e}")
    # Upstream defaults eval to 8; at a 248k vocabulary that is a 52 GiB
    # logits allocation and an OOM at the first evaluation.
    assert targs.per_device_eval_batch_size == 1
    assert targs.per_device_train_batch_size == 1
    assert targs.save_steps == 10
    assert targs.eval_steps == 10


def test_build_training_args_fields(tmp_path):
    args = _args()
    try:
        targs = build_training_args(tmp_path, args)
    except ValueError as e:
        pytest.skip(f"bf16 unsupported on this box: {e}")
    assert str(targs.eval_strategy) in ("steps", "IntervalStrategy.STEPS")
    assert targs.learning_rate == 1e-4
    assert targs.warmup_ratio == 0.03
    assert targs.bf16 is True
    assert targs.metric_for_best_model == "eval_loss"
