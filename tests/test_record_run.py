import json

import pytest

from record_run import render_results, scrub_paths


def test_scrub_paths_replaces_home_and_repo():
    obj = {"run_dir": "/home/alice/repos/segue/.local/runs/r2",
           "args": {"train": "/home/alice/repos/segue/.local/train.jsonl"},
           "n": 3}
    out = scrub_paths(obj, home="/home/alice", repo_root="/home/alice/repos/segue")
    assert out["run_dir"] == "<repo>/.local/runs/r2"
    assert out["args"]["train"] == "<repo>/.local/train.jsonl"
    assert out["n"] == 3


def test_scrub_paths_handles_home_outside_repo_and_lists():
    obj = {"paths": ["/home/alice/models/x", "/opt/shared/y"]}
    out = scrub_paths(obj, home="/home/alice", repo_root="/srv/segue")
    assert out["paths"] == ["~/models/x", "/opt/shared/y"]


def test_scrub_paths_does_not_mutate_input():
    obj = {"p": "/home/alice/x"}
    scrub_paths(obj, home="/home/alice", repo_root="/srv/segue")
    assert obj["p"] == "/home/alice/x"


MANIFEST = {
    "args": {"model": "Qwen/Qwen3.5-9B", "revision": "abc123", "rank": 16,
             "alpha": 32, "epochs": 3, "batch_size": 1, "grad_accum": 16,
             "lr": 0.0001, "seed": 13, "max_length": 16384},
    "env": {"torch": "2.13.0+cu130", "transformers": "5.5.4",
            "device": "NVIDIA GB10"},
    "git": {"segue": "deadbeef"},
    "trainable_params": 43278336,
    "train_loss": 0.9381,
    "final_eval": {"eval_loss": 0.5426},
    "sha256": {"train": "sha256:aaa", "val": "sha256:bbb"},
}
METRICS = {"n": 47, "json_compliance": 1.0, "precision": 0.8333,
           "recall": 0.8696, "f05": 0.8403, "tp": 20, "fp": 4, "fn": 3,
           "noad_windows": 24, "noad_fp": 3, "start_mae": 8.31,
           "end_mae": 7.81}


def test_render_results_contains_headline_numbers():
    md = render_results("r2", MANIFEST, METRICS, None)
    assert "# Run r2" in md
    assert "0.8403" in md and "1.00" in md
    assert "Qwen/Qwen3.5-9B" in md and "abc123" in md
    assert "43,278,336" in md


def test_render_results_flags_small_sample():
    md = render_results("r2", MANIFEST, METRICS, None)
    # 20 tp + 3 fn = 23 truth spans; a reader must not read 0.84 as precise.
    assert "23 truth spans" in md


def test_render_results_without_export_says_so():
    md = render_results("r2", MANIFEST, METRICS, None)
    assert "not exported" in md.lower()


def test_render_results_includes_export_gate():
    exp = {"fixture_matches": [True, True], "logit_max_diff": 0.0,
           "adapter_sha256": {"adapter_model.safetensors": "sha256:ccc"},
           "merged_sha256": {"model.safetensors": "sha256:ddd"}}
    md = render_results("r2", MANIFEST, METRICS, exp)
    assert "2/2" in md
    assert "sha256:ccc" in md
