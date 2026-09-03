import pytest

from spans import GAP_SECONDS, iou, match_spans, merge_gaps


def span(start, end, conf=0.9, **kw):
    base = {"start": start, "end": end, "confidence": conf,
            "category": "sponsor", "reason": "r", "end_text": "buy it now"}
    base.update(kw)
    return base


def test_gap_constant():
    assert GAP_SECONDS == 15.0


def test_merge_gaps_merges_under_gap():
    out = merge_gaps([span(0, 30, 0.8), span(40, 60, 0.95, end_text="later words")])
    assert len(out) == 1
    assert out[0]["start"] == 0 and out[0]["end"] == 60
    assert out[0]["confidence"] == 0.95
    assert out[0]["end_text"] == "later words"


def test_merge_gaps_keeps_first_category_and_reason():
    out = merge_gaps([span(0, 30, category="sponsor", reason="a"),
                      span(40, 60, category="cross_promo", reason="b")])
    assert out[0]["category"] == "sponsor" and out[0]["reason"] == "a"


def test_merge_gaps_exact_gap_does_not_merge():
    out = merge_gaps([span(0, 30), span(45, 60)])
    assert len(out) == 2


def test_merge_gaps_sorts_input():
    out = merge_gaps([span(40, 60), span(0, 30)])
    assert len(out) == 1 and out[0]["start"] == 0


def test_merge_gaps_contained_span():
    out = merge_gaps([span(0, 60), span(10, 20, end_text="inner")])
    assert len(out) == 1
    assert out[0]["end"] == 60 and out[0]["end_text"] == "buy it now"


def test_merge_gaps_empty_and_does_not_mutate():
    assert merge_gaps([]) == []
    original = [span(0, 30), span(5, 40)]
    merge_gaps(original)
    assert original[0]["end"] == 30


def test_iou():
    assert iou((0, 10), (0, 10)) == 1.0
    assert iou((0, 10), (20, 30)) == 0.0
    assert iou((0, 10), (5, 15)) == pytest.approx(1 / 3)


def test_match_spans_greedy_one_to_one():
    preds = [(0.0, 10.0), (0.0, 9.0)]
    truths = [(0.0, 10.0)]
    matches = match_spans(preds, truths)
    assert len(matches) == 1
    assert matches[0][0] == 0 and matches[0][1] == 0


def test_match_spans_threshold():
    assert match_spans([(0.0, 1.0)], [(0.0, 10.0)]) == []


def test_merge_gaps_barrier_blocks_merge():
    spans = [span(0, 30), span(40, 60)]
    assert len(merge_gaps(spans, barriers=[span(31, 39)])) == 2
    assert len(merge_gaps(spans, barriers=[span(70, 80)])) == 1
