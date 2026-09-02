"""Validate every extracted example: schema, holdout, prompt store, sanity.

Usage:
    uv run python tools/validate.py
"""
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    SCHEMA_FILE, iter_examples, load_excluded_feeds, load_holdout, load_prompt,
)
from spans import GAP_SECONDS  # noqa: E402


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


def main():
    validator = Draft202012Validator(json.loads(SCHEMA_FILE.read_text()))
    holdout = load_holdout()
    excluded_feeds = load_excluded_feeds()
    errors = 0
    total = 0
    empty = 0
    checked_prompts = set()
    seen_ids = set()

    for path, line_no, ex in iter_examples():
        total += 1
        where = f"{path}:{line_no}"

        for err in validator.iter_errors(ex):
            print(f"{where}: schema: {err.message}")
            errors += 1

        src = ex.get('source', {})
        key = (src.get('feed'), src.get('episode_id'))
        if key in holdout:
            print(f"{where}: HOLDOUT VIOLATION: {key[0]}/{key[1]}")
            errors += 1
        if src.get('feed') in excluded_feeds:
            print(f"{where}: EXCLUDED FEED: {src.get('feed')}")
            errors += 1

        ex_id = ex.get('id')
        if ex_id in seen_ids:
            print(f"{where}: duplicate id {ex_id}")
            errors += 1
        seen_ids.add(ex_id)

        ref = ex.get('prompt', {}).get('system')
        if ref and ref not in checked_prompts:
            try:
                load_prompt(ref)  # verifies content hash
            except (FileNotFoundError, ValueError) as e:
                print(f"{where}: prompt store: {e}")
                errors += 1
            checked_prompts.add(ref)

        for msg in content_errors(ex):
            print(f"{where}: {msg}")
            errors += 1
        if not ex.get('completion'):
            empty += 1

    print(f"\n{total} examples, {empty} empty ({empty / max(total, 1):.0%}), "
          f"{len(checked_prompts)} prompt template(s), {errors} error(s)")
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
