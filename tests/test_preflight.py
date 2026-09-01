import json

import pytest

from preflight import check, check_disjoint, expected_steps, load_rows


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
    m = tmp_path / "split_manifest.json"
    m.write_text(json.dumps({"ids": {"train": ["a", "b"], "val": ["c"]}}))
    assert check_disjoint(m) is None
    m.write_text(json.dumps({"ids": {"train": ["a", "b"], "val": ["b"]}}))
    assert "overlap" in check_disjoint(m)


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
