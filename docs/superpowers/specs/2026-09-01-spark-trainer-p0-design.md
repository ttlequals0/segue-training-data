# P0 dataset fixes and DGX Spark local trainer

Date: 2026-09-01. Status: approved design, pre-implementation.

## Goal

Fix the P0 problems found in the phase 1 audit, then replace Tinker with a
local Transformers + PEFT + Accelerate trainer running on a DGX Spark, proven
end to end on the cleaned 200-window dataset. Phase 2 data expansion is a
separate later workstream.

## Decisions (fixed)

- Backend: plain HF Trainer + PEFT with an in-repo rendering/masking module.
  TRL SFTTrainer rejected: masking would depend on TRL collator semantics we
  would still have to verify independently.
- Precision: BF16 LoRA. QLoRA rejected; 128 GB unified memory makes 4-bit
  training unnecessary for a 9B model.
- Base model: Qwen/Qwen3.5-9B primary at a pinned revision. Qwen3.5-4B kept
  as a working fallback path. Serving a 9B in the 15 GB budget requires
  int4/AWQ; that validation gate stays in the release phase.
- Span policy: one JSON span is one contiguous ad break. Gaps under 15
  seconds merge. Applied identically to training labels, the prompt rule
  (already states it), and the benchmark scorer, which canonicalizes both
  predictions and truths before IoU matching.
- Outro labels without SHOW SEGMENTS: drop the spans, keep the windows.
- Audio-only-evidence spans: drop the span, keep the window, tag provenance
  so phase 2 can restore them once audio_context is rendered. A committed
  keep-list overrides per span after manual adjudication; it starts with the
  two android-faithful spans whose transcript contains visible ad text.
- train_tinker.py stays untouched as the legacy backend. Local is default.

## Components

### 1. Label contract and schema

- schema/example.schema.json: category enum restricted to sponsor,
  cross_promo, self_promo, interaction. intro/outro/recap legal only when
  provenance records a SHOW SEGMENTS section (none today). reason and
  end_text nonblank. end_text 3 to 5 words. confidence in [0, 1].
- tools/fix_labels.py, one-shot migration over data/examples/:
  - Drop the 4 outro spans; windows kept.
  - Drop spans whose reason matches the audio-only pattern, unless listed in
    the keep-list file (data/keep_spans.json: feed, example id, span index,
    corrected reason). Dropped spans recorded in example provenance under
    dropped_spans with span, rule, and original reason.
  - Recompute every end_text as the last 3 to 5 transcript words before the
    span end, reusing the extractor's clip-boundary logic. Fixes all 12
    empty and 23 out-of-range values uniformly.
  - Merge completion spans with gaps under 15 s (per-break canonicalization).
- tools/validate.py enforces all of the above so the zero-error gate means
  something.

### 2. Dataset build

- tools/build_dataset.py: route each example to train or val first, then
  apply tier weights to train rows only. val.jsonl always full.
- Split manifest written next to outputs: val feeds, example ids per split,
  SHA-256 of both files. Atomic writes (tmp file then rename).

### 3. Benchmark scorer canonicalization (MinusPod repo, separate PR)

- benchmarks/llm/src/benchmark/metrics.py: canonicalize_spans(spans,
  gap=15.0) merges sorted spans with gaps under the threshold; applied to
  predictions and truths before match_predictions. Report regenerated from
  stored raw predictions so all leaderboard rows stay comparable under the
  new policy. Normal MinusPod branch/changelog/PR flow.

### 4. Rendering module (tools/render.py)

- Loads tokenizer at the pinned revision, applies the Qwen chat template
  with enable_thinking=False, builds input_ids and labels masking everything
  except assistant content plus EOS.
- Asserts per example: nonzero label tokens, no truncation at max_length,
  template hash matches the manifest.
- Fail-closed model-family map: unknown family raises instead of silently
  selecting a thinking template.
- Shared by trainer, preflight, eval, and tests.

### 5. Local trainer (tools/train_local.py)

- Plain HF Trainer + PEFT. LoraConfig: task_type=CAUSAL_LM, r=16,
  lora_alpha=32, lora_dropout=0.05, bias="none",
  target_modules="all-linear", modules_to_save=[]. Prints and records
  matched module names and trainable parameter count; asserts the count is
  in the expected range.
- --revision is required (base model commit pin). Defaults: BF16, gradient
  checkpointing, grad clip 1.0, lr 1e-4 linear schedule with 3% warmup,
  fixed seeds, group_by_length.
- Eval set is the feed-held-out val.jsonl for in-run NLL. No random split.
- --run-id names the output dir; --resume requires an existing one and
  restores optimizer state via Trainer resume_from_checkpoint.
- Run manifest written at start and finalized at end: SHA-256 of train, val,
  prompt, rendered template, lockfile; base model commit hash; package
  versions; CUDA, driver, GPU; segue and MinusPod git commits; full args;
  LoRA module list; seeds; checkpoint paths.

### 6. Preflight (tools/preflight.py)

- Checks: CUDA visibility, device capability, BF16 support, Flash Attention
  probe with asserted SDPA fallback, render every example through the
  section 4 assertions, split disjointness, category coverage, one real
  forward/backward/optimizer step, expected step and token counts.
- Fail closed. train_local.py refuses to start without a fresh preflight
  stamp matching the current data hashes.

### 7. Eval and export

- tools/eval_generation.py: greedy decode over val.jsonl with base plus
  adapter, parse, canonicalize per-break, score JSON compliance, span
  precision/recall/F0.5 with IoU matching mirroring the benchmark, no-ad
  false positives, boundary MAE.
- tools/export_local.py: saves the PEFT adapter and a merged BF16 model
  (merge_and_unload), writes checksums, then an equivalence gate: greedy
  generations adapter vs merged on fixed val windows must match exactly;
  logit max-diff reported.

### 8. Spark bring-up (docs/spark-setup.md)

- uv venv with pinned aarch64 CUDA torch wheels; NGC PyTorch container as
  the documented fallback if sm_121 kernels are missing from the wheels.
  Preflight decides, not assumption.
- Runbook: clone, sync data, preflight, smoke LoRA run on the cleaned
  200-window set.

### 9. Testing

- New tests/ (pytest via uv): validator catches each violation class;
  fix_labels transformations including keep-list override; tier weights
  apply to train only; span canonicalization merges; render label masks and
  truncation assertions (CPU, tokenizer only); manifest hashing. TDD during
  implementation. Scorer canonicalization gets its own tests in the
  MinusPod benchmark suite.

## Out of scope (deferred)

- Phase 2 data expansion (backfill, hard negatives, re-windowing,
  audio_context restoration).
- vLLM serving hardening and auth.
- Licensing, model/data cards, release governance.
- Rank sweeps and 4B vs 9B comparisons; those come after the trainer works.

## Verification

- All tests pass locally (CPU-safe subset) and on the Spark (full set).
- tools/validate.py reports zero errors on the migrated dataset.
- Preflight passes on the Spark; one smoke LoRA run completes with the
  manifest fully populated; eval_generation and export equivalence gates
  pass on its output.
