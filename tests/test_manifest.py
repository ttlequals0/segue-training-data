import hashlib
import re

import pytest

from common import REPO_ROOT, sha256_file
from manifest import build_manifest, git_commit


def test_sha256_file(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"hello")
    assert sha256_file(p) == "sha256:" + hashlib.sha256(b"hello").hexdigest()


def test_git_commit_shape():
    out = git_commit(REPO_ROOT)
    assert re.fullmatch(r"[0-9a-f]{40}(\+dirty)?", out)


def test_git_commit_unknown(tmp_path):
    assert git_commit(tmp_path) == "unknown"


def test_build_manifest(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text("{}\n")
    m = build_manifest({"lr": 1e-4}, {"train": p}, {"status": "running"})
    assert m["args"]["lr"] == 1e-4
    assert m["sha256"]["train"].startswith("sha256:")
    assert m["status"] == "running"
    assert "created_at" in m and "git" in m


def test_collect_env_versions():
    pytest.importorskip("torch")
    from manifest import collect_env
    env = collect_env()
    assert {"python", "torch", "transformers", "peft", "accelerate"} <= set(env)
