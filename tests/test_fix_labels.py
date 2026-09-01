from fix_labels import fix_example, parse_segments, recompute_end_text

USER = ("Window transcript:\n"
        "[0.0s - 10.0s] welcome back to the show\n"
        "[10.0s - 20.0s] this episode is sponsored by Acme\n"
        "[20.0s - 30.0s] go to acme example dot com today\n"
        "[30.0s - 40.0s] now back to our guest\n")


def example(completion):
    return {
        "id": "feed-a/ep1/w0",
        "source": {"feed": "feed-a", "episode_id": "ep1", "window": 0},
        "tier": "machine_accepted",
        "prompt": {"system": "sha256:" + "0" * 64, "user": USER,
                   "window_start": 0.0, "window_end": 600.0},
        "completion": completion,
        "provenance": {"reviewed": False, "corrected": False,
                       "category_source": "original",
                       "audio_context_omitted": True,
                       "extracted_at": "2026-09-01T00:00:00Z",
                       "extractor_version": "0.1.0"},
    }


def ad(start, end, category="sponsor", reason="promo read", end_text="x"):
    return {"start": start, "end": end, "confidence": 0.9,
            "category": category, "reason": reason, "end_text": end_text}


def test_parse_segments():
    segs = parse_segments(USER)
    assert len(segs) == 4
    assert segs[1] == (10.0, 20.0, "this episode is sponsored by Acme")


def test_recompute_end_text_takes_trailing_words_inside_span():
    segs = parse_segments(USER)
    assert recompute_end_text(segs, 10.0, 30.0) == "acme example dot com today"


def test_recompute_end_text_short_span():
    segs = parse_segments(USER)
    out = recompute_end_text(segs, 10.0, 20.0)
    assert out == "episode is sponsored by Acme"


def test_outro_span_dropped_window_kept():
    fixed, dropped, fixes = fix_example(example([ad(10, 30, category="outro")]), {})
    assert fixed["completion"] == []
    assert dropped[0]["rule"] == "outro_without_show_segments"
    assert fixed["provenance"]["dropped_spans"][0]["category"] == "outro"
    assert "outro_without_show_segments" in fixes


def test_audio_only_span_dropped_unless_kept():
    reason = "Audio differs across fetches; no other ad signal -- review"
    fixed, dropped, _ = fix_example(example([ad(10, 30, reason=reason)]), {})
    assert fixed["completion"] == [] and dropped[0]["rule"] == "audio_only_evidence"

    keep = {("feed-a/ep1/w0", 10.0): "Ad break with a visible sign-off"}
    fixed, dropped, _ = fix_example(example([ad(10.0, 30, reason=reason)]), keep)
    assert dropped == []
    assert fixed["completion"][0]["reason"] == "Ad break with a visible sign-off"


def test_gaps_merged_and_end_text_recomputed():
    fixed, _, fixes = fix_example(example([ad(0, 12), ad(20, 30)]), {})
    assert len(fixed["completion"]) == 1
    span = fixed["completion"][0]
    assert span["start"] == 0 and span["end"] == 30
    assert span["end_text"] == "x"
    assert "merged_gaps" in fixes
    assert "end_text_recomputed" not in fixes


def test_recompute_end_text_straddling_segment_uses_proportional_prefix():
    segs = parse_segments(USER)
    # segment [20,30] straddles span_end=24: 40% of its 7 words -> first 3
    assert recompute_end_text(segs, 20.0, 24.0) == "go to acme"


def test_end_text_over_5_words_truncated_to_last_5_of_original():
    long_text = "one two three four five six seven"
    fixed, _, fixes = fix_example(
        example([ad(100, 110, end_text=long_text)]), {})
    assert fixed["completion"][0]["end_text"] == "three four five six seven"
    assert "end_text_recomputed" in fixes


def test_empty_end_text_recomputed_from_straddling_segment():
    fixed, _, fixes = fix_example(
        example([ad(20, 24, end_text="")]), {})
    assert fixed["completion"][0]["end_text"] == "go to acme"
    assert "end_text_recomputed" in fixes


def test_clean_example_untouched():
    clean = example([ad(10, 30, end_text="acme example dot com today")])
    fixed, dropped, fixes = fix_example(clean, {})
    assert dropped == [] and fixes == []
    assert "label_fixes" not in fixed["provenance"]
