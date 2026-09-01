import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from validate import content_errors

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schema" / "example.schema.json")
    .read_text())
VALIDATOR = Draft202012Validator(SCHEMA)


def example(completion, provenance_extra=None):
    prov = {"reviewed": False, "corrected": False,
            "category_source": "original", "audio_context_omitted": True,
            "extracted_at": "2026-09-01T00:00:00Z", "extractor_version": "0.1.0"}
    prov.update(provenance_extra or {})
    return {
        "id": "feed-a/ep1/w0",
        "source": {"feed": "feed-a", "episode_id": "ep1", "window": 0},
        "tier": "machine_accepted",
        "prompt": {"system": "sha256:" + "0" * 64,
                   "user": "[0.0s - 600.0s] hello", "window_start": 0.0,
                   "window_end": 600.0},
        "completion": completion,
        "provenance": prov,
    }


def ad(start=0.0, end=30.0, category="sponsor", reason="promo read",
       end_text="go to example.com now"):
    return {"start": start, "end": end, "confidence": 0.9,
            "category": category, "reason": reason, "end_text": end_text}


def schema_errors(ex):
    return [e.message for e in VALIDATOR.iter_errors(ex)]


def test_schema_rejects_outro():
    assert schema_errors(example([ad(category="outro")]))


def test_schema_rejects_blank_end_text_and_reason():
    assert schema_errors(example([ad(end_text="")]))
    assert schema_errors(example([ad(reason="")]))


def test_schema_accepts_label_fixes_and_dropped_spans():
    ex = example([ad()], provenance_extra={
        "label_fixes": ["end_text_recomputed"],
        "dropped_spans": [{"start": 1.0, "end": 2.0, "category": "outro",
                           "reason": "r", "rule": "outro_without_show_segments"}],
    })
    assert schema_errors(ex) == []


def test_content_errors_clean():
    assert content_errors(example([ad()])) == []


def test_content_errors_end_text_word_count():
    assert content_errors(example([ad(end_text="one two three four five six")]))
    assert content_errors(example([ad(end_text="   ")]))


def test_content_errors_unmerged_gap():
    ex = example([ad(0.0, 30.0), ad(40.0, 60.0)])
    assert any("merged" in e for e in content_errors(ex))


def test_content_errors_overlap():
    ex = example([ad(0.0, 30.0), ad(20.0, 60.0)])
    assert any("overlap" in e for e in content_errors(ex))


def test_content_errors_outside_window():
    ex = example([ad(590.0, 650.0)])
    assert any("outside window" in e for e in content_errors(ex))
