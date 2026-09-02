import json

import pytest

from eval_generation import parse_prediction, score


def row(truth):
    return {"messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": json.dumps(truth)},
    ]}


def ad(start, end, conf=0.9):
    return {"start": start, "end": end, "confidence": conf,
            "category": "sponsor", "reason": "r", "end_text": "w x y"}


def test_parse_prediction_strict():
    good = json.dumps([ad(0, 10)])
    assert parse_prediction(good) is not None
    assert parse_prediction("not json") is None
    assert parse_prediction('{"start": 1}') is None
    assert parse_prediction("[1, 2]") is None
    assert parse_prediction('[{"start": "x", "end": 2}]') is None
    assert parse_prediction("[]") == []


def test_score_perfect():
    rows = [row([ad(0, 60)]), row([])]
    preds = [json.dumps([ad(0, 60)]), "[]"]
    s = score(rows, preds)
    assert s["json_compliance"] == 1.0
    assert s["tp"] == 1 and s["fp"] == 0 and s["fn"] == 0
    assert s["noad_windows"] == 1 and s["noad_fp"] == 0
    assert s["f05"] == 1.0


def test_score_merges_per_break_before_matching():
    rows = [row([ad(0, 60)])]
    preds = [json.dumps([ad(0, 25), ad(30, 60)])]
    s = score(rows, preds)
    assert s["tp"] == 1 and s["fp"] == 0 and s["fn"] == 0


def test_score_unparseable_counts_truths_missed():
    rows = [row([ad(0, 60)])]
    s = score(rows, ["garbage"])
    assert s["json_compliance"] == 0.0
    assert s["fn"] == 1 and s["tp"] == 0


def test_score_noad_false_positive():
    rows = [row([])]
    s = score(rows, [json.dumps([ad(0, 10)])])
    assert s["noad_fp"] == 1 and s["fp"] == 1


def test_score_unparseable_noad_window():
    rows = [row([])]
    s = score(rows, ["garbage"])
    assert s["json_compliance"] == 0.0
    assert s["noad_windows"] == 1
    assert s["noad_fp"] == 0 and s["fp"] == 0
    assert s["fn"] == 0


def test_tier_band_matches_published_floors():
    from eval_generation import tier_band
    floors = {"A": 0.760, "B": 0.730, "C": 0.666, "D": 0.546, "E": 0.470,
              "F": 0.349, "G": 0.0}
    assert tier_band(0.861, floors) == "A"
    assert tier_band(0.760, floors) == "A"
    assert tier_band(0.759, floors) == "B"
    assert tier_band(0.686, floors) == "C"
    assert tier_band(0.0, floors) == "G"


def test_tier_band_unordered_floors():
    from eval_generation import tier_band
    # Dict order must not decide the answer.
    floors = {"C": 0.666, "A": 0.760, "G": 0.0, "B": 0.730}
    assert tier_band(0.9, floors) == "A"
    assert tier_band(0.7, floors) == "C"


REPORT_FIXTURE = """
# Benchmark

### Best Accuracy (F0.5 @ IoU >= 0.5)

| Tier | Model | F0.5 | 95% CI |
|---|---|---|---|
| A | `alpha` | 0.861 | +/-0.10 |
| A | `beta` | 0.760 | +/-0.10 |
| B | `gamma` | 0.755 | +/-0.10 |
| B | `delta` | 0.730 | +/-0.10 |
| C | `epsilon` | 0.666 | +/-0.10 |

### Best Free-Tier (F0.5)

| Tier | Model | F0.5 | 95% CI |
|---|---|---|---|
| A | `zeta` | 0.400 | +/-0.10 |
"""


def test_parse_tier_floors_uses_accuracy_table_only():
    from eval_generation import parse_tier_floors
    floors = parse_tier_floors(REPORT_FIXTURE)
    # 0.400 is an A in the free-tier table, scored against its own leader.
    # Letting it in would drag the A floor from 0.760 down to 0.400.
    assert floors == {"A": 0.760, "B": 0.730, "C": 0.666}


def test_parse_tier_floors_rejects_a_report_without_the_table():
    from eval_generation import parse_tier_floors
    with pytest.raises(ValueError, match="Best Accuracy"):
        parse_tier_floors("# Benchmark\n\nnothing here\n")


def test_select_source_requires_exactly_one():
    from eval_generation import select_source
    assert select_source(adapter="a", merged=None, base=False) == "adapter"
    assert select_source(adapter=None, merged="m", base=False) == "merged"
    assert select_source(adapter=None, merged=None, base=True) == "base"
    for bad in ((None, None, False), ("a", "m", False), ("a", None, True)):
        with pytest.raises(SystemExit, match="exactly one"):
            select_source(*bad)
