import json

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
