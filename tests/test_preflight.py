import json

import pytest

from preflight import check, check_disjoint, expected_steps, load_rows
from common import sha256_file


def test_load_rows(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text('{"messages": []}\n\n{"messages": []}\n')
    assert len(load_rows(p)) == 2


def test_load_rows_empty_fails(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(SystemExit):
        load_rows(p)


def test_expected_steps():
    assert expected_steps(156, batch_size=2, grad_accum=8, epochs=3) == 30
    assert expected_steps(16, batch_size=2, grad_accum=8, epochs=1) == 1


def test_check_disjoint(tmp_path):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    train_file.write_text('{"messages": []}\n')
    val_file.write_text('{"messages": []}\n')

    train_sha = sha256_file(train_file)
    val_sha = sha256_file(val_file)

    m = tmp_path / "split_manifest.json"
    m.write_text(json.dumps({
        "ids": {"train": ["a", "b"], "val": ["c"]},
        "sha256": {"train": train_sha, "val": val_sha}
    }))
    assert check_disjoint(m, train_file, val_file) is None

    m.write_text(json.dumps({
        "ids": {"train": ["a", "b"], "val": ["b"]},
        "sha256": {"train": train_sha, "val": val_sha}
    }))
    assert "overlap" in check_disjoint(m, train_file, val_file)


def test_check_disjoint_stale_manifest(tmp_path):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    train_file.write_text('{"messages": []}\n')
    val_file.write_text('{"messages": []}\n')

    m = tmp_path / "split_manifest.json"
    m.write_text(json.dumps({
        "ids": {"train": ["a", "b"], "val": ["c"]},
        "sha256": {"train": "wrong_train_hash", "val": "wrong_val_hash"}
    }))

    error = check_disjoint(m, train_file, val_file)
    assert error is not None
    assert "stale" in error


def test_check_records_system_exit_without_propagating():
    failures = []

    def raises():
        raise SystemExit("boom")

    assert check("x", raises, failures) is None
    assert failures == ["x"]


def test_check_records_value_error():
    failures = []

    def raises():
        raise ValueError("bad")

    assert check("y", raises, failures) is None
    assert failures == ["y"]


def test_check_passes_through_result_on_success():
    failures = []
    assert check("z", lambda: 42, failures) == 42
    assert failures == []


def test_parse_memory_fraction_accepts_valid():
    from preflight import parse_memory_fraction
    assert parse_memory_fraction("0.8") == 0.8
    assert parse_memory_fraction("1.0") == 1.0


def test_parse_memory_fraction_rejects_out_of_range():
    import argparse

    from preflight import parse_memory_fraction
    for bad in ("0", "-0.5", "1.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_memory_fraction(bad)
