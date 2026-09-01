# P0 Dataset Fixes and DGX Spark Local Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the audited P0 dataset contract violations, then add a local Transformers + PEFT + Accelerate LoRA trainer (BF16, Qwen3.5-9B) with preflight, manifests, eval, and export for the DGX Spark.

**Architecture:** A shared `tools/spans.py` defines the per-break span policy once; `tools/fix_labels.py` migrates the committed dataset to it; `tools/render.py` is the single source of truth for chat-template rendering and label masking, consumed by the trainer, preflight, eval, and tests. The MinusPod benchmark scorer gets the same canonicalization in a separate PR so prompt, labels, and scoring all agree.

**Tech Stack:** Python 3.12, uv, pytest, jsonschema, torch >= 2.13, transformers == 5.5.4, peft, accelerate. Two repos: `~/repos/segue-training-data` (tasks 1-12) and `~/repos/MinusPod` (task 13).

**Spec:** `docs/superpowers/specs/2026-09-01-spark-trainer-p0-design.md`

## Global Constraints

- ASCII only in all prose and code. No em dashes and no " -- " dash asides in anything that leaves either repo (commits, PR bodies, docs, comments).
- No Claude attribution or session links in any commit, PR, or artifact, in either repo. Verify before any push: `git log @{upstream}..HEAD --format=%B | grep -i claude` returns nothing (use `origin/main..HEAD` on a new branch).
- segue-training-data work happens on branch `spark-trainer`. MinusPod work happens on branch `fix/benchmark-per-break-scoring`; never commit to MinusPod main.
- TDD: write the failing test first, watch it fail, implement, watch it pass, commit.
- Run segue tests with `uv run pytest tests/ -v` from the segue repo root. Tests that need torch/transformers guard with `pytest.importorskip`; tests that download the Qwen tokenizer are marked `tokenizer` and skip cleanly offline.
- Do not modify `tools/train_tinker.py` (legacy backend, stays as-is) or any LLM prompt string literals under `prompts/`.
- The span policy constant is 15.0 seconds, strict less-than (`gap < 15.0` merges). Defined once per repo: `tools/spans.py:GAP_SECONDS` (segue) and `metrics.CANONICAL_GAP_SECONDS` (MinusPod).
- Base model default: `Qwen/Qwen3.5-9B`. `--revision` is always required for training/eval/export; never train against a floating revision.
- MinusPod PR bodies: no segue feed slugs, episode IDs, or instance identifiers; run the /humanizer skill on the body before posting; grep the body for " -- " before posting.

---

### Task 1: Test scaffolding and the span policy module

**Files:**
- Create: `tests/__init__.py` (empty), `tests/conftest.py`, `tests/test_spans.py`
- Create: `tools/spans.py`
- Modify: `pyproject.toml` (add dev dependency group)

**Interfaces:**
- Produces: `spans.GAP_SECONDS: float = 15.0`; `spans.merge_gaps(spans: list[dict], gap: float = GAP_SECONDS) -> list[dict]` (dicts with `start`, `end`, `confidence` keys, optional others; merged span keeps first span's `category`/`reason`, later-ending span's `end_text`, max `confidence`); `spans.iou(a: tuple[float, float], b: tuple[float, float]) -> float`; `spans.match_spans(predictions: list[tuple], truths: list[tuple], threshold: float = 0.5) -> list[tuple[int, int, float]]` (pred_index, truth_index, iou; greedy one-to-one, mirrors the MinusPod benchmark).

- [ ] **Step 1: Add pytest and conftest**

Append to `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    "pytest>=8.3",
]
```

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
```

Run: `uv sync && uv run pytest tests/ -v`
Expected: "no tests ran" (collection works, zero tests).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_spans.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_spans.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spans'`.

- [ ] **Step 4: Implement `tools/spans.py`**

```python
"""Span policy: one span is one contiguous ad break (sub-15s gaps merge)."""

GAP_SECONDS = 15.0


def merge_gaps(spans, gap=GAP_SECONDS):
    """Merge dict spans whose gap is under `gap` seconds."""
    if not spans:
        return []
    ordered = sorted((dict(s) for s in spans),
                     key=lambda s: (s["start"], s["end"]))
    out = [ordered[0]]
    for s in ordered[1:]:
        cur = out[-1]
        if s["start"] - cur["end"] < gap:
            if s["end"] > cur["end"]:
                cur["end"] = s["end"]
                if "end_text" in s:
                    cur["end_text"] = s["end_text"]
            cur["confidence"] = max(cur["confidence"], s["confidence"])
        else:
            out.append(s)
    return out


def iou(a, b):
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if overlap == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - overlap
    return overlap / union if union > 0 else 0.0


def match_spans(predictions, truths, threshold=0.5):
    """Greedy one-to-one IoU matching; mirrors the MinusPod benchmark scorer."""
    pairs = []
    for pi, p in enumerate(predictions):
        for ti, t in enumerate(truths):
            score = iou(p, t)
            if score >= threshold:
                pairs.append((score, pi, ti))
    pairs.sort(key=lambda x: x[0], reverse=True)
    used_p, used_t, matches = set(), set(), []
    for score, pi, ti in pairs:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matches.append((pi, ti, score))
    return matches
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_spans.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/ tools/spans.py
git commit -m "Add span policy module and test scaffolding"
```

---

### Task 2: Tighten the schema and validator

**Files:**
- Modify: `schema/example.schema.json`
- Modify: `tools/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: `spans.GAP_SECONDS`.
- Produces: `validate.content_errors(ex: dict) -> list[str]` (content checks beyond the JSON schema; empty list means clean). The schema now rejects categories outside the four allowed, blank `reason`/`end_text`, and permits `provenance.label_fixes` (list of strings) and `provenance.dropped_spans`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validate.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validate.py -v`
Expected: FAIL (`content_errors` does not exist; schema tests for outro/blank fail because the current schema allows them).

- [ ] **Step 3: Update the schema**

In `schema/example.schema.json`:

1. Replace the completion category enum line with:

```json
"enum": ["sponsor", "cross_promo", "self_promo", "interaction"]
```

2. Add `"minLength": 1` to both `reason` and `end_text`:

```json
"reason": {"type": "string", "minLength": 1},
"end_text": {"type": "string", "minLength": 1}
```

3. In `provenance.properties`, add:

```json
"label_fixes": {"type": "array", "items": {"type": "string"}},
"dropped_spans": {
  "type": "array",
  "items": {
    "type": "object",
    "required": ["start", "end", "category", "reason", "rule"],
    "additionalProperties": false,
    "properties": {
      "start": {"type": "number"},
      "end": {"type": "number"},
      "category": {"type": "string"},
      "reason": {"type": "string"},
      "rule": {"type": "string",
               "enum": ["outro_without_show_segments", "audio_only_evidence"]}
    }
  }
}
```

- [ ] **Step 4: Add `content_errors` to `tools/validate.py`**

Add after the imports (which gain `from spans import GAP_SECONDS`):

```python
def content_errors(ex):
    """Content checks beyond the JSON schema. Empty list means clean."""
    errs = []
    w = ex.get('prompt', {})
    prev_end = None
    for i, ad in enumerate(ex.get('completion', [])):
        if not (w.get('window_start', 0) <= ad['start'] < ad['end']
                <= w.get('window_end', float('inf')) + 0.01):
            errs.append(f"ad {ad['start']}-{ad['end']} outside window "
                        f"{w.get('window_start')}-{w.get('window_end')}")
        n_words = len(ad['end_text'].split())
        if not 1 <= n_words <= 5:
            errs.append(f"ad {i}: end_text must be 1-5 words, got {n_words}: "
                        f"{ad['end_text']!r}")
        if not ad['reason'].strip():
            errs.append(f"ad {i}: blank reason")
        if prev_end is not None:
            if ad['start'] < prev_end:
                errs.append(f"ad {i}: overlaps previous span")
            elif ad['start'] - prev_end < GAP_SECONDS:
                errs.append(f"ad {i}: gap under {GAP_SECONDS}s from previous "
                            f"span; per-break policy requires them merged")
        prev_end = ad['end']
    return errs
```

In `main()`, replace the existing window-bounds loop (the `for ad in ex.get('completion', [])` block) with:

```python
        for msg in content_errors(ex):
            print(f"{where}: {msg}")
            errors += 1
```

Note: the prompt's "3-5 words" rule for `end_text` is relaxed to 1-5 in
validation because a very short span can contain fewer than 3 transcript
words. Task 12 records this in design.md.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_validate.py tests/test_spans.py -v`
Expected: all PASS.

- [ ] **Step 6: Confirm the validator now fails on the uncorrected dataset**

Run: `uv run python tools/validate.py; echo "exit=$?"`
Expected: nonzero exit with errors for outro categories, blank end_text, and word counts. This is correct at this point; task 4 fixes the data. Do not "fix" the validator to pass here.

- [ ] **Step 7: Commit**

```bash
git add schema/example.schema.json tools/validate.py tests/test_validate.py
git commit -m "Tighten schema and validator to the per-break label contract"
```

---

### Task 3: The label migration tool

**Files:**
- Create: `tools/fix_labels.py`
- Test: `tests/test_fix_labels.py`

**Interfaces:**
- Consumes: `spans.merge_gaps`, `spans.GAP_SECONDS`, `common.EXAMPLES_DIR`, `common.REPO_ROOT`.
- Produces: `fix_labels.fix_example(ex: dict, keep: dict[tuple[str, float], str]) -> tuple[dict, list[dict], list[str]]` returning (fixed example, dropped span dicts each with a `rule` key, list of fix tags applied); `fix_labels.parse_segments(user_prompt: str) -> list[tuple[float, float, str]]`; `fix_labels.recompute_end_text(segments, span_start: float, span_end: float, n: int = 5) -> str`. Keep-list file `data/keep_spans.json`: JSON array of `{"id": "<example id>", "start": <span start>, "reason": "<corrected reason>"}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fix_labels.py`:

```python
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
    assert span["end_text"] == "acme example dot com today"
    assert "merged_gaps" in fixes and "end_text_recomputed" in fixes


def test_clean_example_untouched():
    clean = example([ad(10, 30, end_text="acme example dot com today")])
    fixed, dropped, fixes = fix_example(clean, {})
    assert dropped == [] and fixes == []
    assert "label_fixes" not in fixed["provenance"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fix_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fix_labels'`.

- [ ] **Step 3: Implement `tools/fix_labels.py`**

```python
"""One-shot label migration to the per-break output contract.

Drops outro spans (no phase 1 prompt has SHOW SEGMENTS), drops
audio-only-evidence spans unless kept by data/keep_spans.json, merges
sub-15s gaps, and recomputes every end_text from the window transcript.

Usage:
    uv run python tools/fix_labels.py --dry-run   # review drops first
    uv run python tools/fix_labels.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXAMPLES_DIR, REPO_ROOT  # noqa: E402
from spans import GAP_SECONDS, merge_gaps  # noqa: E402

KEEP_FILE = REPO_ROOT / 'data' / 'keep_spans.json'
ALLOWED_CATEGORIES = {'sponsor', 'cross_promo', 'self_promo', 'interaction'}
AUDIO_ONLY = re.compile(r'no other ad signal', re.IGNORECASE)
SEGMENT_LINE = re.compile(r'^\[(\d+(?:\.\d+)?)s - (\d+(?:\.\d+)?)s\] (.*)$')


def parse_segments(user_prompt):
    segs = []
    for line in user_prompt.splitlines():
        m = SEGMENT_LINE.match(line.strip())
        if m:
            segs.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    return segs


def recompute_end_text(segments, span_start, span_end, n=5):
    """Last `n` transcript words from segments ending inside the span."""
    words = []
    for s, e, text in segments:
        if e <= span_start or s >= span_end:
            continue
        if e <= span_end + 2.0:
            words.extend(text.split())
    if not words:
        words = [w for s, e, text in segments
                 if s < span_end and e > span_start for w in text.split()]
    return ' '.join(words[-n:])


def load_keep_list():
    if not KEEP_FILE.exists():
        return {}
    entries = json.loads(KEEP_FILE.read_text(encoding='utf-8'))
    return {(e['id'], float(e['start'])): e['reason'] for e in entries}


def fix_example(ex, keep):
    """Returns (fixed example, dropped spans, fix tags applied)."""
    dropped, spans, fixes = [], [], []
    for ad in ex['completion']:
        if ad['category'] not in ALLOWED_CATEGORIES:
            dropped.append({**ad, 'rule': 'outro_without_show_segments'})
            continue
        if AUDIO_ONLY.search(ad['reason']):
            new_reason = keep.get((ex['id'], float(ad['start'])))
            if new_reason is None:
                dropped.append({**ad, 'rule': 'audio_only_evidence'})
                continue
            ad = {**ad, 'reason': new_reason}
            if 'reason_corrected' not in fixes:
                fixes.append('reason_corrected')
        spans.append(dict(ad))
    fixes = list(dict.fromkeys([d['rule'] for d in dropped] + fixes))
    n_before = len(spans)
    spans = merge_gaps(spans, GAP_SECONDS)
    if len(spans) != n_before:
        fixes.append('merged_gaps')
    segments = parse_segments(ex['prompt']['user'])
    for ad in spans:
        new_text = recompute_end_text(segments, ad['start'], ad['end'])
        if new_text and new_text != ad['end_text']:
            ad['end_text'] = new_text
            if 'end_text_recomputed' not in fixes:
                fixes.append('end_text_recomputed')
    out = dict(ex)
    out['completion'] = spans
    if dropped or fixes:
        prov = dict(out['provenance'])
        if dropped:
            prov['dropped_spans'] = [
                {'start': d['start'], 'end': d['end'], 'category': d['category'],
                 'reason': d['reason'], 'rule': d['rule']} for d in dropped]
        if fixes:
            prov['label_fixes'] = fixes
        out['provenance'] = prov
    return out, dropped, fixes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    keep = load_keep_list()
    stats = {'examples': 0, 'changed': 0, 'dropped': 0}
    for path in sorted(EXAMPLES_DIR.rglob('*.jsonl')):
        lines_out, file_changed = [], False
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            stats['examples'] += 1
            fixed, dropped, fixes = fix_example(ex, keep)
            for d in dropped:
                segs = parse_segments(ex['prompt']['user'])
                tail = recompute_end_text(segs, d['start'], d['end'])
                print(f"DROP {ex['id']} [{d['start']}-{d['end']}] "
                      f"{d['rule']} reason={d['reason']!r} "
                      f"transcript_tail={tail!r}")
            stats['dropped'] += len(dropped)
            if fixes or dropped:
                stats['changed'] += 1
                file_changed = True
            lines_out.append(json.dumps(fixed, ensure_ascii=False))
        if file_changed and not args.dry_run:
            tmp = path.with_suffix('.jsonl.tmp')
            tmp.write_text('\n'.join(lines_out) + '\n', encoding='utf-8')
            tmp.replace(path)
    mode = 'dry-run' if args.dry_run else 'applied'
    print(f"{mode}: {stats['examples']} examples, {stats['changed']} changed, "
          f"{stats['dropped']} spans dropped, {len(keep)} keep-list entries")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fix_labels.py -v`
Expected: all PASS. If `test_clean_example_untouched` fails because the recompute produces different trailing words than the test expects, fix the test's `end_text` to the exact recompute output rather than loosening the tool.

- [ ] **Step 5: Commit**

```bash
git add tools/fix_labels.py tests/test_fix_labels.py
git commit -m "Add fix_labels migration tool for the per-break contract"
```

---

### Task 4: Run the migration on the committed dataset

**Files:**
- Create: `data/keep_spans.json`
- Modify: `data/examples/**/*.jsonl` (via the tool, never by hand)

**Interfaces:**
- Consumes: `tools/fix_labels.py`, `tools/validate.py`.

- [ ] **Step 1: Dry run and review every drop**

Run: `uv run python tools/fix_labels.py --dry-run | tee /tmp/fix-labels-dry.txt`
Expected: DROP lines for the 4 outro spans and the audio-only spans, each with its `transcript_tail`.

- [ ] **Step 2: Build the keep-list**

Adjudication criterion (from the approved design): keep an audio-only span
only when its `transcript_tail` shows promotional language (a sponsor read,
a network sign-off, a call to action). Known candidates from the audit: the
two android-faithful spans ending "visiting acast.com". Check the
security-now span (tail "KQED. Tap to listen now.") against the same
criterion and include it if it qualifies.

Create `data/keep_spans.json` with one entry per kept span, using the exact
`id` and `start` printed by the dry run:

```json
[
  {
    "id": "<id from dry run>",
    "start": 5217.8,
    "reason": "Injected network ad read ending with an acast.com call to action"
  }
]
```

Write each `reason` from the transcript evidence shown in the dry run; do
not reuse the audio-only wording.

- [ ] **Step 3: Re-run dry run, then apply**

Run: `uv run python tools/fix_labels.py --dry-run` and confirm the keep-list
entries no longer print as DROP. Then: `uv run python tools/fix_labels.py`
Expected: "applied" summary; changed count > 0.

- [ ] **Step 4: Validate the migrated dataset**

Run: `uv run python tools/validate.py`
Expected: `0 error(s)`, exit 0. If errors remain (for example an `end_text`
recompute produced 0 words), fix the tool or the keep-list and re-run the
tool; never hand-edit the data files.

- [ ] **Step 5: Spot-check the numbers against the audit**

Run:

```bash
uv run python - <<'EOF'
import json, sys
sys.path.insert(0, 'tools')
from common import iter_examples
ads = empty = dropped = 0
bad_words = 0
for _, _, ex in iter_examples():
    spans = ex['completion']
    ads += len(spans)
    empty += not spans
    dropped += len(ex['provenance'].get('dropped_spans', []))
    bad_words += sum(1 for a in spans if not 1 <= len(a['end_text'].split()) <= 5)
print(f"ads={ads} empty={empty} dropped={dropped} bad_end_text={bad_words}")
EOF
```

Expected: `bad_end_text=0`; `dropped` equals the dry-run drop count; `ads`
is below the pre-migration 75 by (drops + merges).

- [ ] **Step 6: Commit the migration**

```bash
git add data/
git commit -m "Migrate labels to the per-break contract; add keep-list"
```

---

### Task 5: Fix the dataset build (tier weights, manifest, atomic writes)

**Files:**
- Modify: `tools/build_dataset.py`
- Modify: `tools/common.py` (add `sha256_file`)
- Test: `tests/test_build_dataset.py`

**Interfaces:**
- Consumes: `common.sha256_file(path: Path) -> str` (format `sha256:<hex>`), added here.
- Produces: `build_dataset.split_examples(examples: list[dict], val_feeds: set[str], weights: dict[str, float], rng: random.Random) -> tuple[list[dict], list[dict], int]` returning (train, val, downsampled_count); `<out-dir>/split_manifest.json` with keys `val_feeds`, `seed`, `tier_weights`, `counts`, `ids` (per split), `sha256` (per output file).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_dataset.py`:

```python
import random

from build_dataset import split_examples


def ex(feed, tier="machine_accepted", n=0):
    return {"id": f"{feed}/ep/w{n}", "tier": tier,
            "source": {"feed": feed, "episode_id": "ep", "window": n},
            "prompt": {"system": "sha256:" + "0" * 64, "user": "u",
                       "window_start": 0.0, "window_end": 1.0},
            "completion": []}


def test_val_feeds_route_to_val():
    train, val, down = split_examples(
        [ex("a"), ex("b")], {"b"}, {}, random.Random(13))
    assert [e["id"] for e in train] == ["a/ep/w0"]
    assert [e["id"] for e in val] == ["b/ep/w0"]
    assert down == 0


def test_tier_weights_never_touch_val():
    examples = [ex("a", n=i) for i in range(50)] + [ex("b", n=i) for i in range(50)]
    train, val, down = split_examples(
        examples, {"b"}, {"machine_accepted": 0.0}, random.Random(13))
    assert train == [] and down == 50
    assert len(val) == 50


def test_split_is_disjoint_and_complete():
    examples = [ex("a", n=i) for i in range(10)] + [ex("b", n=i) for i in range(10)]
    train, val, down = split_examples(examples, {"b"}, {}, random.Random(13))
    ids = {e["id"] for e in train} | {e["id"] for e in val}
    assert len(ids) == 20 and down == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_build_dataset.py -v`
Expected: FAIL with ImportError (`split_examples` does not exist).

- [ ] **Step 3: Implement**

Add to `tools/common.py` (after `sha256_text`):

```python
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()
```

In `tools/build_dataset.py`, add `split_examples` and rework `main()`.
Replace the whole `with (out_dir / 'train.jsonl')...` block (and the counts
dict) with:

```python
def split_examples(examples, val_feeds, weights, rng):
    """Route first, then downsample train only; val is never sampled."""
    train, val, downsampled = [], [], 0
    for ex in examples:
        if ex['source']['feed'] in val_feeds:
            val.append(ex)
        elif rng.random() >= weights.get(ex['tier'], 1.0):
            downsampled += 1
        else:
            train.append(ex)
    return train, val, downsampled
```

and in `main()`:

```python
    examples = [ex for _, _, ex in iter_examples()]
    seen_feeds = {ex['source']['feed'] for ex in examples}
    missing = val_feeds - seen_feeds
    if missing:
        raise SystemExit(f"val feeds not found in examples: {sorted(missing)}")

    train, val, downsampled = split_examples(examples, val_feeds, weights, rng)
    system_cache = {}
    splits = {'train': train, 'val': val}
    for name, rows in splits.items():
        tmp = out_dir / f'{name}.jsonl.tmp'
        with tmp.open('w', encoding='utf-8') as f:
            for ex in rows:
                f.write(json.dumps(to_chat(ex, system_cache),
                                   ensure_ascii=False) + '\n')
        tmp.replace(out_dir / f'{name}.jsonl')

    manifest = {
        'val_feeds': sorted(val_feeds),
        'seed': args.seed,
        'tier_weights': weights,
        'counts': {name: len(rows) for name, rows in splits.items()},
        'downsampled': downsampled,
        'ids': {name: [ex['id'] for ex in rows] for name, rows in splits.items()},
        'sha256': {name: sha256_file(out_dir / f'{name}.jsonl') for name in splits},
    }
    tmp = out_dir / 'split_manifest.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    tmp.replace(out_dir / 'split_manifest.json')

    print(f"train: {len(train)}  val: {len(val)}  downsampled: {downsampled}")
    print(f"wrote train.jsonl, val.jsonl, split_manifest.json in {out_dir}")
```

Update the import line in `build_dataset.py` to include `sha256_file`:

```python
from common import REPO_ROOT, iter_examples, load_prompt, sha256_file  # noqa: E402
```

- [ ] **Step 4: Run tests, then a real build**

Run: `uv run pytest tests/ -v`
Expected: all PASS.

Run a real build using the phase 1 val feeds (read them from
`runs/20260830-162400-results.md` or pick the same three whole feeds used
before; they are listed in the results file's split description):

```bash
uv run python tools/build_dataset.py --val-feeds <feed1>,<feed2>,<feed3> --out-dir .local
cat .local/split_manifest.json | head -20
```

Expected: counts match `validate.py` totals; manifest present.

- [ ] **Step 5: Commit**

```bash
git add tools/build_dataset.py tools/common.py tests/test_build_dataset.py
git commit -m "Route split before tier sampling; emit split manifest atomically"
```

---

### Task 6: The run-manifest module

**Files:**
- Create: `tools/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `common.sha256_file`, `common.REPO_ROOT`.
- Produces: `manifest.collect_env() -> dict` (python/platform/torch/transformers/peft/accelerate versions, cuda fields when available); `manifest.git_commit(repo_dir: Path) -> str` (40-hex sha, `+dirty` suffix when the tree is dirty, `unknown` on failure); `manifest.build_manifest(args: dict, files: dict[str, str | Path], extra: dict) -> dict` with keys `args`, `sha256` (per file), `env`, `git` (segue + minuspod when present), `created_at`, plus `extra` merged at top level.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_manifest.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'manifest'`.

- [ ] **Step 3: Implement `tools/manifest.py`**

```python
"""Run manifest: everything needed to reproduce a training run."""
import datetime
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, sha256_file  # noqa: E402


def collect_env():
    import accelerate
    import peft
    import torch
    import transformers
    env = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'peft': peft.__version__,
        'accelerate': accelerate.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env['cuda'] = torch.version.cuda
        env['device'] = torch.cuda.get_device_name(0)
        env['capability'] = '.'.join(
            map(str, torch.cuda.get_device_capability(0)))
    return env


def git_commit(repo_dir):
    try:
        head = subprocess.run(
            ['git', '-C', str(repo_dir), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ['git', '-C', str(repo_dir), 'status', '--porcelain'],
            capture_output=True, text=True, check=True).stdout.strip()
        return head + ('+dirty' if dirty else '')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def build_manifest(args, files, extra):
    git = {'segue': git_commit(REPO_ROOT)}
    minuspod = REPO_ROOT.parent / 'MinusPod'
    if minuspod.is_dir():
        git['minuspod'] = git_commit(minuspod)
    m = {
        'created_at': datetime.datetime.now(datetime.timezone.utc)
                      .strftime('%Y-%m-%dT%H:%M:%SZ'),
        'args': dict(args),
        'sha256': {name: sha256_file(p) for name, p in files.items()},
        'git': git,
    }
    m.update(extra)
    return m
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: PASS (the `collect_env` test skips until task 7 installs torch; that is fine).

- [ ] **Step 5: Commit**

```bash
git add tools/manifest.py tests/test_manifest.py
git commit -m "Add run-manifest helpers: env, git, and file hashes"
```

---

### Task 7: The rendering and masking module, plus the local training extra

**Files:**
- Create: `tools/render.py`
- Modify: `pyproject.toml` (add `local` extra)
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (transformers tokenizer only).
- Produces: `render.MODEL_FAMILIES: dict` (currently `{"qwen3.5": {"kwargs": {"enable_thinking": False}, "suffix": "<|im_end|>\n"}}`); `render.family_for(model_name: str) -> str` (raises `render.UnsupportedModelFamily` for unmapped names); `render.template_hash(tokenizer) -> str`; `render.render_texts(tokenizer, messages: list[dict], model_name: str) -> tuple[str, str]` returning (generation prefix, full text = prefix + assistant content + suffix); `render.encode_example(tokenizer, messages, model_name, max_length: int) -> dict` with `input_ids`, `labels` (prefix masked to -100), `length`; raises ValueError on truncation, zero-loss targets, or prefix/tokenization misalignment.

- [ ] **Step 1: Add the local extra**

In `pyproject.toml` under `[project.optional-dependencies]` add:

```toml
local = [
    "torch>=2.13",
    "transformers==5.5.4",
    "peft>=0.17",
    "accelerate>=1.6",
]
```

Run: `uv sync --extra local`
Expected: resolves and installs (CPU torch on this workstation is fine; the
Spark install path is task 12's doc).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_render.py`:

```python
import json
import os

import pytest

transformers = pytest.importorskip("transformers")

import render  # noqa: E402

TEST_MODEL = os.environ.get("SEGUE_TEST_TOKENIZER", "Qwen/Qwen3.5-4B")


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(TEST_MODEL)
    except OSError as e:
        pytest.skip(f"tokenizer unavailable offline: {e}")


MESSAGES = [
    {"role": "system", "content": "You detect ads. Answer with a JSON array."},
    {"role": "user", "content": "[0.0s - 10.0s] buy stuff at example.com"},
    {"role": "assistant", "content": json.dumps(
        [{"start": 0.0, "end": 10.0, "confidence": 0.9, "category": "sponsor",
          "reason": "promo read", "end_text": "at example.com"}],
        separators=(",", ":"))},
]


def test_family_for():
    assert render.family_for("Qwen/Qwen3.5-9B") == "qwen3.5"
    with pytest.raises(render.UnsupportedModelFamily):
        render.family_for("Qwen/Qwen3.8-27B")
    with pytest.raises(render.UnsupportedModelFamily):
        render.family_for("meta-llama/Llama-3-8B")


@pytest.mark.tokenizer
def test_prefix_is_nonthinking(tokenizer):
    prefix, full = render.render_texts(tokenizer, MESSAGES, TEST_MODEL)
    assert prefix.endswith("<think>\n\n</think>\n\n")
    assert full.startswith(prefix)
    assert full.endswith("<|im_end|>\n")


@pytest.mark.tokenizer
def test_labels_mask_exactly_the_prefix(tokenizer):
    enc = render.encode_example(tokenizer, MESSAGES, TEST_MODEL, 16384)
    n_masked = sum(1 for t in enc["labels"] if t == -100)
    n_target = len(enc["labels"]) - n_masked
    assert n_target > 0
    target_ids = [t for t in enc["labels"] if t != -100]
    decoded = tokenizer.decode(target_ids)
    assert decoded == MESSAGES[2]["content"] + "<|im_end|>\n"
    assert enc["input_ids"][:n_masked] == tokenizer(
        render.render_texts(tokenizer, MESSAGES, TEST_MODEL)[0],
        add_special_tokens=False)["input_ids"]


@pytest.mark.tokenizer
def test_truncation_raises(tokenizer):
    with pytest.raises(ValueError, match="tokens"):
        render.encode_example(tokenizer, MESSAGES, TEST_MODEL, 16)
```

Register the mark in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["tokenizer: needs the Qwen tokenizer (downloads from HF hub)"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'render'`.

- [ ] **Step 4: Implement `tools/render.py`**

```python
"""Render chat examples through the exact Qwen non-thinking template.

Single source of truth for tokenization and label masking; the trainer,
preflight, eval, and tests all consume this module. The full text is
composed as prefix + assistant content + suffix so the label boundary is
deterministic and inspectable; assertions fail closed on template drift.
"""
import hashlib


class UnsupportedModelFamily(SystemExit):
    pass


MODEL_FAMILIES = {
    'qwen3.5': {
        'kwargs': {'enable_thinking': False},
        'suffix': '<|im_end|>\n',
        'nonthinking_tail': '<think>\n\n</think>\n\n',
    },
}


def family_for(model_name):
    name = model_name.lower()
    for family in MODEL_FAMILIES:
        if family in name:
            return family
    raise UnsupportedModelFamily(
        f'no non-thinking template mapping for {model_name}; '
        f'known families: {sorted(MODEL_FAMILIES)}')


def template_hash(tokenizer):
    return 'sha256:' + hashlib.sha256(
        tokenizer.chat_template.encode('utf-8')).hexdigest()


def render_texts(tokenizer, messages, model_name):
    fam = MODEL_FAMILIES[family_for(model_name)]
    if messages[-1]['role'] != 'assistant':
        raise ValueError('last message must be the assistant completion')
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True,
        **fam['kwargs'])
    if not prefix.endswith(fam['nonthinking_tail']):
        raise ValueError(
            'generation prefix does not end with the non-thinking tail; '
            'template drift, refusing to guess label boundaries')
    return prefix, prefix + messages[-1]['content'] + fam['suffix']


def encode_example(tokenizer, messages, model_name, max_length):
    prefix, full = render_texts(tokenizer, messages, model_name)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    full_ids = tokenizer(full, add_special_tokens=False)['input_ids']
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError('prefix tokenization is not a prefix of the full '
                         'tokenization; label mask would be misaligned')
    if len(full_ids) > max_length:
        raise ValueError(f'example is {len(full_ids)} tokens, over {max_length}')
    if len(full_ids) == len(prefix_ids):
        raise ValueError('no assistant tokens carry loss')
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    return {'input_ids': full_ids, 'labels': labels, 'length': len(full_ids)}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_render.py -v`
Expected: all PASS (tokenizer tests download ~10 MB once, then cache). If
`test_prefix_is_nonthinking` fails on the tail assertion, inspect the actual
rendered prefix with `tokenizer.apply_chat_template(..., tokenize=False)`
and update `nonthinking_tail` to the template's real non-thinking tail; the
audit measured `<|im_start|>assistant\n<think>\n\n</think>\n\n` on
transformers 5.5.4.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tools/render.py tests/test_render.py
git commit -m "Add fail-closed chat rendering with assistant-only label masks"
```

---

### Task 8: Preflight

**Files:**
- Create: `tools/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `render.encode_example`, `render.template_hash`, `common.sha256_file`, `manifest.collect_env`.
- Produces: `preflight.load_rows(path) -> list[dict]` (chat rows); `preflight.check_disjoint(manifest_path: Path) -> str | None` (error string or None); `preflight.render_all(tokenizer, rows, model_name, max_length) -> dict` (stats: n, token min/median/max, categories, empty_fraction; raises on any bad example); `preflight.expected_steps(n: int, batch_size: int, grad_accum: int, epochs: int) -> int`; stamp file `.local/preflight.json` with keys `train_sha256`, `val_sha256`, `model`, `revision`, `attn`, `max_length`, `template_hash`, `passed_at`, `stats`. Exit 0 only when every check passes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_preflight.py`:

```python
import json

import pytest

from preflight import check_disjoint, expected_steps, load_rows


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_preflight.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tools/preflight.py`**

```python
"""Preflight gate: refuse to train until data, renderer, and device check out.

Writes .local/preflight.json; train_local.py requires a stamp whose hashes
match its inputs.

Usage:
    uv run python tools/preflight.py --model Qwen/Qwen3.5-9B --revision <sha>
    uv run python tools/preflight.py ... --device cpu --skip-model-step
"""
import argparse
import datetime
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render  # noqa: E402
from common import REPO_ROOT, sha256_file  # noqa: E402

STAMP = REPO_ROOT / '.local' / 'preflight.json'


def load_rows(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f'{path}: no examples')
    return rows


def check_disjoint(manifest_path):
    if not Path(manifest_path).exists():
        return f'{manifest_path} missing; rebuild the dataset'
    ids = json.loads(Path(manifest_path).read_text())['ids']
    overlap = set(ids['train']) & set(ids['val'])
    if overlap:
        return f'train/val overlap: {sorted(overlap)[:5]}'
    return None


def render_all(tokenizer, rows, model_name, max_length):
    lengths, categories, empty = [], {}, 0
    for i, row in enumerate(rows):
        enc = render.encode_example(tokenizer, row['messages'],
                                    model_name, max_length)
        lengths.append(enc['length'])
        ads = json.loads(row['messages'][2]['content'])
        if not ads:
            empty += 1
        for ad in ads:
            categories[ad['category']] = categories.get(ad['category'], 0) + 1
    return {
        'n': len(rows),
        'tokens_min': min(lengths),
        'tokens_median': statistics.median(lengths),
        'tokens_max': max(lengths),
        'categories': categories,
        'empty_fraction': round(empty / len(rows), 3),
    }


def expected_steps(n, batch_size, grad_accum, epochs):
    return max(1, math.ceil(n / (batch_size * grad_accum))) * epochs


def optimizer_step_smoke(model_name, revision, attn, tokenizer, rows,
                         model_max_length):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=torch.bfloat16,
        attn_implementation=attn, device_map='cuda')
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, bias='none', target_modules='all-linear'))
    model.enable_input_require_grads()
    longest = max(rows, key=lambda r: len(r['messages'][1]['content']))
    enc = render.encode_example(tokenizer, longest['messages'],
                                model_name, model_max_length)
    ids = torch.tensor([enc['input_ids']], device='cuda')
    labels = torch.tensor([enc['labels']], device='cuda')
    opt = torch.optim.AdamW((p for p in model.parameters()
                             if p.requires_grad), lr=1e-4)
    loss = model(input_ids=ids, labels=labels).loss
    loss.backward()
    opt.step()
    peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30
    return {'loss': float(loss), 'peak_memory_gb': round(peak_gb, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', default=str(REPO_ROOT / '.local' / 'train.jsonl'))
    ap.add_argument('--val', default=str(REPO_ROOT / '.local' / 'val.jsonl'))
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--revision', required=True)
    ap.add_argument('--max-length', type=int, default=16384)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--skip-model-step', action='store_true')
    args = ap.parse_args()

    failures = []

    def check(name, fn):
        try:
            result = fn()
            print(f'PASS {name}' + (f': {result}' if result else ''))
            return result
        except Exception as e:
            print(f'FAIL {name}: {e}')
            failures.append(name)
            return None

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              revision=args.revision)
    train_rows = check('train file', lambda: load_rows(args.train)) or []
    val_rows = check('val file', lambda: load_rows(args.val)) or []
    manifest_path = Path(args.train).parent / 'split_manifest.json'

    def disjoint():
        err = check_disjoint(manifest_path)
        if err:
            raise SystemExit(err)

    check('split disjoint', disjoint)
    train_stats = check('render train', lambda: render_all(
        tokenizer, train_rows, args.model, args.max_length))
    val_stats = check('render val', lambda: render_all(
        tokenizer, val_rows, args.model, args.max_length))
    steps = expected_steps(len(train_rows), args.batch_size,
                           args.grad_accum, args.epochs)
    print(f'INFO expected optimizer steps: {steps}')

    attn = 'sdpa'
    if args.device == 'cuda':
        import torch
        check('cuda visible', lambda: torch.cuda.get_device_name(0))

        def bf16():
            if not torch.cuda.is_bf16_supported():
                raise SystemExit('bf16 unsupported')

        check('bf16 supported', bf16)
        try:
            import flash_attn  # noqa: F401
            attn = 'flash_attention_2'
        except ImportError:
            pass
        print(f'INFO attention implementation: {attn}')
        if not args.skip_model_step:
            check('optimizer step', lambda: optimizer_step_smoke(
                args.model, args.revision, attn, tokenizer, train_rows,
                args.max_length))

    if failures:
        raise SystemExit(f'preflight FAILED: {failures}')

    stamp = {
        'train_sha256': sha256_file(args.train),
        'val_sha256': sha256_file(args.val),
        'model': args.model,
        'revision': args.revision,
        'attn': attn,
        'max_length': args.max_length,
        'template_hash': render.template_hash(tokenizer),
        'passed_at': datetime.datetime.now(datetime.timezone.utc)
                     .strftime('%Y-%m-%dT%H:%M:%SZ'),
        'stats': {'train': train_stats, 'val': val_stats,
                  'expected_steps': steps},
    }
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(json.dumps(stamp, indent=2) + '\n')
    print(f'preflight PASSED; stamp written to {STAMP}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests, then a CPU preflight against the real split**

Run: `uv run pytest tests/test_preflight.py -v`
Expected: PASS.

Pin the base model revision once and reuse it everywhere from here on:

```bash
REV=$(uv run python -c "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/Qwen3.5-9B').sha)")
echo $REV
uv run python tools/preflight.py --revision $REV --device cpu --skip-model-step
```

Expected: every data check PASS, stamp written. The tokenizer downloads;
the model does not. Record `$REV` in the task report; tasks 9-11 use it.

- [ ] **Step 5: Commit**

```bash
git add tools/preflight.py tests/test_preflight.py
git commit -m "Add fail-closed preflight with data, render, and device checks"
```

---

### Task 9: The local trainer

**Files:**
- Create: `tools/train_local.py`
- Test: `tests/test_train_local.py`

**Interfaces:**
- Consumes: `render.encode_example`, `render.template_hash`, `manifest.build_manifest`, `manifest.collect_env`, `common.sha256_file`, preflight stamp `.local/preflight.json`.
- Produces: `train_local.build_lora_config(r, alpha, dropout) -> peft.LoraConfig`; `train_local.expected_lora_params(model, r) -> int` (sum of `r * (in + out)` over `nn.Linear` modules excluding `lm_head`); `train_local.make_collator(pad_id) -> callable`; `train_local.require_stamp(train_path, val_path, model, revision) -> dict`; run directory `.local/runs/<run-id>/` (checkpoints, `adapter/`), manifest `runs/<run-id>.json` (committed to git).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_local.py`:

```python
import json

import pytest

torch = pytest.importorskip("torch")

from train_local import (  # noqa: E402
    build_lora_config, expected_lora_params, make_collator, require_stamp,
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
        require_stamp(train, val, "m", "rev")
    from common import sha256_file
    stamp.write_text(json.dumps({
        "train_sha256": sha256_file(train), "val_sha256": sha256_file(val),
        "model": "m", "revision": "rev", "attn": "sdpa",
        "max_length": 16384}))
    assert require_stamp(train, val, "m", "rev")["attn"] == "sdpa"
    train.write_text('{"changed": 1}\n')
    with pytest.raises(SystemExit, match="preflight"):
        require_stamp(train, val, "m", "rev")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train_local.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train_local'`.

- [ ] **Step 3: Implement `tools/train_local.py`**

```python
"""LoRA-train Segue locally (DGX Spark) with Transformers + PEFT.

Requires a passing preflight stamp (tools/preflight.py) for the same data,
model, and revision.

Usage:
    uv run python tools/train_local.py --run-id r1 --revision <commit-sha>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as manifest_mod  # noqa: E402
import render  # noqa: E402
from common import REPO_ROOT, sha256_file  # noqa: E402

STAMP = REPO_ROOT / '.local' / 'preflight.json'
DEFAULT_MODEL = 'Qwen/Qwen3.5-9B'


def build_lora_config(r, alpha, dropout):
    from peft import LoraConfig, TaskType
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha,
                      lora_dropout=dropout, bias='none',
                      target_modules='all-linear')


def expected_lora_params(model, r):
    """Params all-linear LoRA adds: r * (in + out) per Linear, minus lm_head."""
    import torch.nn as nn
    total = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and 'lm_head' not in name:
            total += r * (mod.in_features + mod.out_features)
    return total


def make_collator(pad_id):
    def collate(features):
        import torch
        width = max(f['length'] for f in features)
        ids, labels, mask = [], [], []
        for f in features:
            pad = width - f['length']
            ids.append(f['input_ids'] + [pad_id] * pad)
            labels.append(f['labels'] + [-100] * pad)
            mask.append([1] * f['length'] + [0] * pad)
        return {'input_ids': torch.tensor(ids),
                'labels': torch.tensor(labels),
                'attention_mask': torch.tensor(mask)}
    return collate


def require_stamp(train_path, val_path, model, revision):
    if not STAMP.exists():
        raise SystemExit('no preflight stamp; run tools/preflight.py first')
    stamp = json.loads(STAMP.read_text())
    expect = {'train_sha256': sha256_file(train_path),
              'val_sha256': sha256_file(val_path),
              'model': model, 'revision': revision}
    for key, want in expect.items():
        if stamp.get(key) != want:
            raise SystemExit(
                f'preflight stamp mismatch on {key}; rerun tools/preflight.py')
    return stamp


def load_rows(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', default=str(REPO_ROOT / '.local' / 'train.jsonl'))
    ap.add_argument('--val', default=str(REPO_ROOT / '.local' / 'val.jsonl'))
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--resume', action='store_true',
                    help='resume from the latest checkpoint in the run dir')
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=0.05)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--max-length', type=int, default=16384)
    ap.add_argument('--seed', type=int, default=13)
    args = ap.parse_args()

    stamp = require_stamp(args.train, args.val, args.model, args.revision)
    run_dir = REPO_ROOT / '.local' / 'runs' / args.run_id
    if run_dir.exists() and not args.resume:
        raise SystemExit(f'{run_dir} exists; pass --resume or a new --run-id')
    if args.resume and not run_dir.exists():
        raise SystemExit(f'--resume but {run_dir} does not exist')
    run_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              revision=args.revision)
    if render.template_hash(tokenizer) != stamp['template_hash']:
        raise SystemExit('chat template changed since preflight; rerun it')
    train_enc = [render.encode_example(tokenizer, r['messages'], args.model,
                                       args.max_length)
                 for r in load_rows(args.train)]
    val_enc = [render.encode_example(tokenizer, r['messages'], args.model,
                                     args.max_length)
               for r in load_rows(args.val)]

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        attn_implementation=stamp['attn'])
    expected = expected_lora_params(model, args.rank)
    model = get_peft_model(model, build_lora_config(
        args.rank, args.alpha, args.dropout))
    model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    matched = sorted({n.split('.lora_')[0].rsplit('.', 1)[-1]
                      for n, _ in model.named_parameters() if 'lora_' in n})
    print(f'LoRA modules: {matched}')
    print(f'trainable params: {trainable:,} (expected {expected:,})')
    if trainable != expected:
        raise SystemExit('trainable parameter count does not match all-linear '
                         'expectation; inspect target_modules')

    run_manifest = manifest_mod.build_manifest(
        vars(args),
        {'train': args.train, 'val': args.val,
         'lockfile': REPO_ROOT / 'uv.lock'},
        {'status': 'running', 'env': manifest_mod.collect_env(),
         'template_hash': stamp['template_hash'], 'attn': stamp['attn'],
         'lora_modules': matched, 'trainable_params': trainable,
         'run_dir': str(run_dir)})
    manifest_path = REPO_ROOT / 'runs' / f'{args.run_id}.json'
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + '\n')

    targs = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type='linear',
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        logging_steps=1,
        eval_strategy='steps',
        eval_steps=10,
        save_steps=20,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_enc,
                      eval_dataset=val_enc,
                      data_collator=make_collator(
                          tokenizer.pad_token_id or tokenizer.eos_token_id))
    result = trainer.train(resume_from_checkpoint=args.resume)
    final_eval = trainer.evaluate()

    adapter_dir = run_dir / 'adapter'
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    run_manifest.update({
        'status': 'completed',
        'train_loss': result.training_loss,
        'final_eval': final_eval,
        'best_checkpoint': trainer.state.best_model_checkpoint,
        'adapter_sha256': {p.name: sha256_file(p)
                           for p in sorted(adapter_dir.iterdir())
                           if p.is_file()},
    })
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + '\n')
    print(f'done; adapter in {adapter_dir}, manifest in {manifest_path}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train_local.py -v`
Expected: all PASS. Also run the full suite: `uv run pytest tests/ -v`.

Note: the end-to-end training run happens on the Spark (task 12 runbook);
this task's deliverable is the tested tool. Do not attempt to load the 9B
on this workstation.

Deviation from the spec, recorded here: the spec listed `group_by_length`;
the trainer uses plain padding instead because length-grouped sampling
needs a datasets.Dataset column and the padding waste at 200 examples is
negligible. Revisit if the phase 2 dataset makes step time matter.

- [ ] **Step 5: Commit**

```bash
git add tools/train_local.py tests/test_train_local.py
git commit -m "Add local BF16 LoRA trainer with manifest and preflight gate"
```

---

### Task 10: Generation eval

**Files:**
- Create: `tools/eval_generation.py`
- Test: `tests/test_eval_generation.py`

**Interfaces:**
- Consumes: `spans.merge_gaps`, `spans.match_spans`, `render.render_texts`.
- Produces: `eval_generation.parse_prediction(text: str) -> list[dict] | None` (strict bare-JSON-array parse; None on any violation); `eval_generation.score(rows: list[dict], predictions: list[str]) -> dict` with keys `n`, `json_compliance`, `precision`, `recall`, `f05`, `tp`, `fp`, `fn`, `noad_windows`, `noad_fp`, `start_mae`, `end_mae`; report file `.local/eval-<run-id>.json`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_generation.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_generation.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tools/eval_generation.py`**

```python
"""Generation eval on the feed-held-out val set.

Scores JSON compliance, per-break span P/R/F0.5 at IoU >= 0.5 (mirrors the
MinusPod benchmark scorer), no-ad false positives, and boundary MAE.

Usage:
    uv run python tools/eval_generation.py --run-id r1 --revision <sha> \
        --adapter .local/runs/r1/adapter
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render  # noqa: E402
from common import REPO_ROOT  # noqa: E402
from spans import match_spans, merge_gaps  # noqa: E402


def parse_prediction(text):
    """Strict parse: a bare JSON array of span objects, else None."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    for ad in data:
        if not isinstance(ad, dict) or not {'start', 'end'} <= set(ad):
            return None
        if not all(isinstance(ad[k], (int, float)) and not isinstance(ad[k], bool)
                   for k in ('start', 'end')):
            return None
    return data


def _ranges(ads):
    normed = [{'start': a['start'], 'end': a['end'],
               'confidence': float(a.get('confidence', 0.0))} for a in ads]
    return [(s['start'], s['end']) for s in merge_gaps(normed)]


def score(rows, predictions):
    tp = fp = fn = parsed = 0
    noad_windows = noad_fp = 0
    start_deltas, end_deltas = [], []
    for row, pred in zip(rows, predictions):
        truth = _ranges(json.loads(row['messages'][2]['content']))
        ads = parse_prediction(pred)
        if ads is None:
            fn += len(truth)
            continue
        parsed += 1
        spans = _ranges(ads)
        if not truth:
            noad_windows += 1
            noad_fp += len(spans)
            fp += len(spans)
            continue
        matches = match_spans(spans, truth)
        tp += len(matches)
        fp += len(spans) - len(matches)
        fn += len(truth) - len(matches)
        for pi, ti, _ in matches:
            start_deltas.append(abs(spans[pi][0] - truth[ti][0]))
            end_deltas.append(abs(spans[pi][1] - truth[ti][1]))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denom = 0.25 * precision + recall
    return {
        'n': len(rows),
        'json_compliance': round(parsed / len(rows), 4) if rows else 0.0,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f05': round(1.25 * precision * recall / denom, 4) if denom else 0.0,
        'tp': tp, 'fp': fp, 'fn': fn,
        'noad_windows': noad_windows, 'noad_fp': noad_fp,
        'start_mae': round(statistics.fmean(start_deltas), 2)
                     if start_deltas else None,
        'end_mae': round(statistics.fmean(end_deltas), 2)
                   if end_deltas else None,
    }


def generate_all(model, tokenizer, rows, model_name, max_new_tokens=1024):
    import torch
    outputs = []
    for row in rows:
        prefix, _ = render.render_texts(tokenizer, row['messages'], model_name)
        ids = tokenizer(prefix, add_special_tokens=False,
                        return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, do_sample=False,
                                 max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id
                                 or tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids['input_ids'].shape[1]:],
                                skip_special_tokens=True)
        outputs.append(text.strip())
    return outputs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--val', default=str(REPO_ROOT / '.local' / 'val.jsonl'))
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--revision', required=True)
    ap.add_argument('--adapter', help='PEFT adapter dir (omit with --merged)')
    ap.add_argument('--merged', help='merged model dir (omit with --adapter)')
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    if bool(args.adapter) == bool(args.merged):
        raise SystemExit('pass exactly one of --adapter or --merged')

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(args.val, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.merged:
        tokenizer = AutoTokenizer.from_pretrained(args.merged)
        model = AutoModelForCausalLM.from_pretrained(
            args.merged, dtype=torch.bfloat16, device_map='auto')
    else:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.model,
                                                  revision=args.revision)
        base = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16,
            device_map='auto')
        model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    predictions = generate_all(model, tokenizer, rows, args.model)
    result = score(rows, predictions)
    out = REPO_ROOT / '.local' / f'eval-{args.run_id}.json'
    out.write_text(json.dumps(
        {'args': {k: v for k, v in vars(args).items()},
         'metrics': result}, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print(f'written to {out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_generation.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/eval_generation.py tests/test_eval_generation.py
git commit -m "Add held-out generation eval with per-break scoring"
```

---

### Task 11: Export with equivalence gate

**Files:**
- Create: `tools/export_local.py`
- Test: `tests/test_export_local.py`

**Interfaces:**
- Consumes: `render.render_texts`, `common.sha256_file`, `manifest.build_manifest`.
- Produces: `export_local.checksum_dir(path: Path) -> dict[str, str]`; `export_local.generations_match(model_a, model_b, input_ids, max_new_tokens=128) -> bool` (greedy token-id equality); output layout `<out>/adapter/`, `<out>/merged/`, `<out>/export_manifest.json` (checksums, revision, fixture results, logit max diff). Exits 1 when any fixture generation differs between adapter and merged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_export_local.py`:

```python
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from export_local import checksum_dir, generations_match  # noqa: E402


def tiny_pair():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, vocab_size=128)
    peft_model = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
        target_modules="all-linear"))
    for _, p in peft_model.named_parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.02)
    import copy
    merged = copy.deepcopy(peft_model).merge_and_unload()
    return peft_model.eval(), merged.eval()


def test_merged_matches_adapter_generations():
    a, b = tiny_pair()
    ids = torch.randint(0, 128, (1, 8))
    assert generations_match(a, b, ids, max_new_tokens=16)


def test_generations_match_detects_difference():
    a, _ = tiny_pair()
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(99)
    other = LlamaForCausalLM(LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128)).eval()
    ids = torch.randint(0, 128, (1, 8))
    assert not generations_match(a, other, ids, max_new_tokens=16)


def test_checksum_dir(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y")
    sums = checksum_dir(tmp_path)
    assert set(sums) == {"a.bin", "sub/b.bin"}
    assert all(v.startswith("sha256:") for v in sums.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_export_local.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tools/export_local.py`**

```python
"""Export the trained adapter and a merged BF16 model, then verify equivalence.

The gate: greedy generations from base+adapter and from the merged model
must be token-identical on fixture windows before the export is accepted.

Usage:
    uv run python tools/export_local.py --run-id r1 --revision <sha> \
        --adapter .local/runs/r1/adapter
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as manifest_mod  # noqa: E402
import render  # noqa: E402
from common import REPO_ROOT, sha256_file  # noqa: E402


def checksum_dir(path):
    path = Path(path)
    return {str(p.relative_to(path)): sha256_file(p)
            for p in sorted(path.rglob('*')) if p.is_file()}


def generations_match(model_a, model_b, input_ids, max_new_tokens=128):
    import torch
    outs = []
    for model in (model_a, model_b):
        with torch.no_grad():
            out = model.generate(input_ids=input_ids.to(model.device),
                                 do_sample=False,
                                 max_new_tokens=max_new_tokens)
        outs.append(out[0].tolist())
    return outs[0] == outs[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--revision', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--fixtures', default=str(REPO_ROOT / '.local' / 'val.jsonl'))
    ap.add_argument('--n-fixtures', type=int, default=8)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = Path(args.out or REPO_ROOT / '.local' / 'export' / args.run_id)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              revision=args.revision)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        device_map='auto')
    adapter_model = PeftModel.from_pretrained(base, args.adapter).eval()

    out.mkdir(parents=True, exist_ok=True)
    adapter_out = out / 'adapter'
    if adapter_out.exists():
        shutil.rmtree(adapter_out)
    shutil.copytree(args.adapter, adapter_out)

    with open(args.fixtures, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    fixtures = rows[:args.n_fixtures]
    fixture_ids = []
    for row in fixtures:
        prefix, _ = render.render_texts(tokenizer, row['messages'], args.model)
        fixture_ids.append(tokenizer(prefix, add_special_tokens=False,
                                     return_tensors='pt')['input_ids'])

    with torch.no_grad():
        ref_logits = adapter_model(fixture_ids[0].to(adapter_model.device)).logits

    merged = adapter_model.merge_and_unload()
    merged_out = out / 'merged'
    merged.save_pretrained(str(merged_out), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_out))
    del adapter_model, merged, base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    reloaded = AutoModelForCausalLM.from_pretrained(
        str(merged_out), dtype=torch.bfloat16, device_map='auto').eval()
    base2 = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        device_map='auto')
    adapter2 = PeftModel.from_pretrained(base2, str(adapter_out)).eval()

    with torch.no_grad():
        merged_logits = reloaded(fixture_ids[0].to(reloaded.device)).logits
    logit_max_diff = float((ref_logits.cpu() - merged_logits.cpu())
                           .abs().max())
    results = [generations_match(adapter2, reloaded, ids)
               for ids in fixture_ids]

    export_manifest = manifest_mod.build_manifest(
        vars(args), {},
        {'adapter_sha256': checksum_dir(adapter_out),
         'merged_sha256': checksum_dir(merged_out),
         'fixture_matches': results,
         'logit_max_diff': logit_max_diff})
    (out / 'export_manifest.json').write_text(
        json.dumps(export_manifest, indent=2) + '\n')
    print(f'fixtures matched: {sum(results)}/{len(results)}, '
          f'logit max diff {logit_max_diff:.2e}')
    if not all(results):
        raise SystemExit('EQUIVALENCE FAILED: merged model diverges from '
                         'base+adapter; do not publish this export')
    print(f'export verified; artifacts in {out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_export_local.py -v`
Expected: all PASS (tiny models, CPU, no downloads). Note: BF16 merge
rounding can in principle flip a greedy token; if the Spark run ever fails
the gate with a tiny `logit_max_diff`, report it rather than loosening the
gate, per the spec.

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest tests/ -v`
Expected: all PASS.

```bash
git add tools/export_local.py tests/test_export_local.py
git commit -m "Add adapter+merged export with generation equivalence gate"
```

---

### Task 12: Documentation: Spark setup, design.md, README

**Files:**
- Create: `docs/spark-setup.md`
- Modify: `docs/design.md`, `README.md`

**Interfaces:**
- Consumes: everything built in tasks 1-11 (documents it).

- [ ] **Step 1: Write `docs/spark-setup.md`**

```markdown
# DGX Spark setup and first run

The DGX Spark is an aarch64 (Grace) host with a Blackwell GPU (sm_121) and
128 GB of unified memory. Two environment paths; preflight decides which
works, in this order.

## Path 1: uv venv with CUDA wheels (preferred)

    git clone <this repo> && cd segue-training-data
    uv sync --extra local
    uv pip install torch --index-url https://download.pytorch.org/whl/cu130

PyPI aarch64 torch wheels are CPU-only; the cu-series index carries the
CUDA aarch64 builds. Then verify:

    uv run python -c "import torch; print(torch.cuda.get_device_name(0),
    torch.cuda.get_device_capability(0), torch.cuda.is_bf16_supported())"

If this fails or preflight's optimizer-step check reports missing sm_121
kernels, use path 2.

## Path 2: NGC PyTorch container (fallback)

Run the current NGC PyTorch release for DGX Spark with the repo mounted:

    docker run --gpus all -it --rm -v $PWD:/work -w /work \
        nvcr.io/nvidia/pytorch:<current-release>-py3
    pip install transformers==5.5.4 peft accelerate jsonschema

## First run

    export HF_HOME=/path/with/space
    hf download Qwen/Qwen3.5-9B --revision <pinned-sha>

    uv run python tools/preflight.py --revision <pinned-sha>
    uv run python tools/train_local.py --run-id r1 --revision <pinned-sha>
    uv run python tools/eval_generation.py --run-id r1 --revision <pinned-sha> \
        --adapter .local/runs/r1/adapter
    uv run python tools/export_local.py --run-id r1 --revision <pinned-sha> \
        --adapter .local/runs/r1/adapter

Preflight must pass before training; the trainer refuses a stale or missing
stamp. The run manifest lands in runs/<run-id>.json and is committed.

## Sizing

9B BF16 weights are ~18 GB; LoRA training with gradient checkpointing and
16k-token windows fits comfortably in unified memory. Preflight's optimizer
step prints measured peak memory for the longest window.
```

Replace `<current-release>` guidance only if a known-good tag is confirmed
on the Spark; do not invent a tag.

- [ ] **Step 2: Update `docs/design.md`**

Make these targeted edits (keep everything else):

1. In "Design decisions", base model bullet: state Qwen3.5-9B is the primary
   training target on local hardware with Qwen3.5-4B kept as the fallback,
   and that serving a 9B in the 15 GB budget requires int4/AWQ, validated
   before release.
2. Add a design decision bullet: "One output span is one contiguous ad
   break; gaps under 15 seconds merge. Enforced in the training labels
   (tools/fix_labels.py), the validator, the eval scorer, and the MinusPod
   benchmark scorer (canonicalized before IoU matching). end_text is
   validated at 1-5 words: the prompt asks for 3-5 but a short span can
   contain fewer transcript words."
3. In "Approach", note the local trainer (tools/train_local.py, Transformers
   + PEFT, BF16 LoRA, DGX Spark) is now the default backend and
   train_tinker.py is legacy.
4. In "Status", add: dataset migrated to the per-break contract (spans
   dropped/merged per the audit), validator tightened, local trainer with
   preflight/manifest/eval/export landed. Note the phase 1 F0.5 0.686 row
   predates scorer canonicalization and is not directly comparable to
   scores produced after it.

- [ ] **Step 3: Update `README.md`**

Add the new tools to the tool listing (fix_labels, preflight, train_local,
eval_generation, export_local; one line each) and a "Local training (DGX
Spark)" section pointing at `docs/spark-setup.md`. Mark train_tinker.py as
the legacy backend.

- [ ] **Step 4: Check prose hygiene**

Run from the repo root:

```bash
grep -rn " -- " docs/spark-setup.md docs/design.md README.md; \
grep -rnP '[^\x00-\x7F]' docs/spark-setup.md docs/design.md README.md
```

Expected: no output from either grep.

- [ ] **Step 5: Commit**

```bash
git add docs/spark-setup.md docs/design.md README.md
git commit -m "Document the Spark setup, span policy, and local-first training"
```

---

### Task 13: MinusPod benchmark scorer canonicalization (separate repo and PR)

**Files (all under `~/repos/MinusPod`):**
- Modify: `benchmarks/llm/src/benchmark/metrics.py`
- Modify: `benchmarks/llm/src/benchmark/report/aggregate.py:326-365`
- Modify: `benchmarks/llm/tests/test_metrics.py`
- Modify: `version.py`, `openapi.yaml`, `CHANGELOG.md`
- Regenerate: `benchmarks/llm/results/report.md` + `report_assets`

**Interfaces:**
- Produces: `metrics.CANONICAL_GAP_SECONDS: float = 15.0`; `metrics.canonicalize_spans(spans: list[tuple[float, float]], *, gap: float = CANONICAL_GAP_SECONDS) -> list[tuple[float, float]]`; `metrics.canonicalize_ads(ads: list[dict], *, gap: float = CANONICAL_GAP_SECONDS) -> list[dict]` (dict-level merge on `start`/`end`, keeps max numeric `confidence`, first dict's other fields, so `flat_ads` stays index-aligned with the span list for calibration).

- [ ] **Step 1: Branch**

```bash
cd ~/repos/MinusPod && git checkout main && git pull && \
git checkout -b fix/benchmark-per-break-scoring
```

- [ ] **Step 2: Write the failing tests**

Append to `benchmarks/llm/tests/test_metrics.py`:

```python
def test_canonicalize_spans_merges_under_gap():
    from benchmark.metrics import canonicalize_spans
    assert canonicalize_spans([(0.0, 30.0), (40.0, 60.0)]) == [(0.0, 60.0)]


def test_canonicalize_spans_exact_gap_does_not_merge():
    from benchmark.metrics import canonicalize_spans
    assert canonicalize_spans([(0.0, 30.0), (45.0, 60.0)]) == [
        (0.0, 30.0), (45.0, 60.0)]


def test_canonicalize_spans_sorts_and_handles_containment():
    from benchmark.metrics import canonicalize_spans
    assert canonicalize_spans([(40.0, 60.0), (0.0, 30.0)]) == [(0.0, 60.0)]
    assert canonicalize_spans([(0.0, 60.0), (10.0, 20.0)]) == [(0.0, 60.0)]
    assert canonicalize_spans([]) == []


def test_canonicalize_ads_keeps_max_confidence_and_alignment():
    from benchmark.metrics import canonicalize_ads
    ads = [{"start": 0.0, "end": 30.0, "confidence": 0.8, "category": "sponsor"},
           {"start": 40.0, "end": 60.0, "confidence": 0.95},
           {"start": 200.0, "end": 230.0, "confidence": None}]
    out = canonicalize_ads(ads)
    assert len(out) == 2
    assert out[0]["start"] == 0.0 and out[0]["end"] == 60.0
    assert out[0]["confidence"] == 0.95
    assert out[0]["category"] == "sponsor"
    assert out[1]["confidence"] is None


def test_per_spot_predictions_match_per_break_truth_after_canon():
    from benchmark.metrics import canonicalize_spans, match_predictions
    preds = canonicalize_spans([(0.0, 25.0), (30.0, 60.0)])
    truths = canonicalize_spans([(0.0, 60.0)])
    r = match_predictions(preds, truths, threshold=0.5)
    assert r.true_positives == 1 and r.false_negatives == 0
```

Run: `cd benchmarks/llm && uv run pytest tests/test_metrics.py -v`
Expected: new tests FAIL with ImportError.

- [ ] **Step 3: Implement in `metrics.py`**

Add after the `iou` function:

```python
CANONICAL_GAP_SECONDS = 15.0


def canonicalize_spans(
    spans: list[tuple[float, float]], *, gap: float = CANONICAL_GAP_SECONDS
) -> list[tuple[float, float]]:
    """Merge spans separated by less than `gap` seconds.

    One span = one contiguous ad break, the detection prompt's merge rule.
    Applied to predictions and truths alike before matching.
    """
    if not spans:
        return []
    ordered = sorted(spans)
    out = [list(ordered[0])]
    for s, e in ordered[1:]:
        if s - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def canonicalize_ads(
    ads: list[dict], *, gap: float = CANONICAL_GAP_SECONDS
) -> list[dict]:
    """Dict-level merge on start/end so ads stay aligned with their spans."""
    if not ads:
        return []
    ordered = sorted((dict(a) for a in ads),
                     key=lambda a: (a["start"], a["end"]))
    out = [ordered[0]]
    for a in ordered[1:]:
        cur = out[-1]
        if a["start"] - cur["end"] < gap:
            cur["end"] = max(cur["end"], a["end"])
            ca, cb = cur.get("confidence"), a.get("confidence")
            if isinstance(cb, (int, float)) and (
                    not isinstance(ca, (int, float)) or cb > ca):
                cur["confidence"] = cb
        else:
            out.append(a)
    return out
```

- [ ] **Step 4: Wire into `aggregate.py`**

In the scoring loop (around line 326), immediately after
`flat_ads: list[dict] = parsing.deduplicate_window_ads(norm_ads)`:

```python
        flat_ads = metrics.canonicalize_ads(flat_ads)
        flat_preds: list[tuple[float, float]] = [(_start(ad), _end(ad)) for ad in flat_ads]
```

(replacing the existing `flat_preds` line). In the else-branch, replace

```python
            truth_ranges = [(ad.start, ad.end) for ad in ep.truth.ads]
```

with

```python
            truth_ranges = metrics.canonicalize_spans(
                [(ad.start, ad.end) for ad in ep.truth.ads])
```

and replace the detection-bucket loop (which iterates `ep.truth.ads` and
would misalign with canonical truth indices) with:

```python
            for ti, (ts, te) in enumerate(truth_ranges):
                hit = ti in matched_truth_idxs
                detection_buckets[model]["length"][_length_bucket(te - ts)].append(hit)
                detection_buckets[model]["position"][_position_bucket(ts, duration)].append(hit)
```

The no-ad path needs no change (no truths; every prediction is an FP either
way).

- [ ] **Step 5: Run the benchmark test suite**

Run: `cd benchmarks/llm && uv run pytest tests/ -v`
Expected: all PASS, including pre-existing tests. If an existing aggregate
or report test asserts pre-canonicalization counts, update that test's
expectation to the per-break result and say so in the PR body.

- [ ] **Step 6: Regenerate the report from stored raw calls**

```bash
cd benchmarks/llm && uv run benchmark report --help
```

Pass the same frozen prompt snapshot the stored report footer names (check
the current `results/report.md` footer for the snapshot file), for example:

```bash
uv run benchmark report --prompt-snapshot prompts/<frozen-2026-08-snapshot>
```

Expected: `results/report.md` and `report_assets` regenerate offline from
`results/raw/calls.jsonl`. Review the diff: scores may shift for every
model; that is the point. Sanity-check one model's row moved in the
expected direction (recall up for models that split breaks, precision up
for models that merged).

- [ ] **Step 7: Version, changelog**

- `version.py`: `2.88.3` -> `2.89.0` (scoring semantics change; minor).
- `openapi.yaml`: match the version field.
- `CHANGELOG.md` under a new `## [2.89.0]` section, `### Changed`:

```markdown
- Benchmark scorer now canonicalizes predictions and ground truth to
  per-break spans (gaps under 15 seconds merged) before IoU matching,
  matching the detection prompt's merge rule. Report regenerated from the
  stored raw calls; per-model scores shift accordingly and are not
  comparable to pre-2.89.0 rows.
```

- [ ] **Step 8: Quality gates and PR**

1. Run `/simplify`, then `/code-review` on the branch (repo rule; fix all
   findings).
2. Audit branch-added comments per the CLAUDE.md comment rules.
3. Write the PR title `Benchmark: score per-break via span canonicalization
   (2.89.0)` and a body describing the policy, the wiring points, and the
   report regeneration. No segue feed names, no instance identifiers, no
   " -- ", no session links.
4. Invoke the /humanizer skill on the PR body before posting (Skill tool
   invocation, not a mental pass).
5. Verify: `git log origin/main..HEAD --format=%B | grep -i claude` returns
   nothing.
6. Push the branch and open the PR with `gh pr create`. Do NOT merge; merge
   authorization is the user's, and the release flow follows their call.

---

## Final verification (after all tasks)

- [ ] segue repo: `uv run pytest tests/ -v` all green; `uv run python tools/validate.py` exits 0 with 0 errors.
- [ ] segue repo: `git log origin/main..HEAD --format=%B | grep -i claude` returns nothing; no " -- " or non-ASCII in new docs.
- [ ] MinusPod: benchmark suite green, PR open, not merged.
- [ ] Spark (when configured): preflight passes, smoke run `r1` completes, `runs/r1.json` manifest fully populated, eval report written, export equivalence gate passes. These steps wait for the hardware; everything else must not.
