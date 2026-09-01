# Segue design and status

Segue is an open-weight ad-detection model for
[MinusPod](https://github.com/ttlequals0/MinusPod). The goal is a model small
enough to run in 15 GB of VRAM that can replace the cloud LLM in MinusPod's
detection stage: find ad segments in podcast transcripts, categorize each one
(sponsor, cross_promo, self_promo, interaction), and answer in the exact JSON
format the pipeline parses. Weights will be published on Hugging Face.

## Approach

Thin vertical slice first, then widen. Phase 1 proves every link of the chain
(extraction, training, serving, scoring) on a small dataset before the bulk of
the effort goes into data breadth. The local trainer (`tools/train_local.py`,
Transformers + PEFT, BF16 LoRA, run on a DGX Spark) is now the default
training backend. `tools/train_tinker.py`, which trains on
[Tinker](https://thinkingmachines.ai/tinker/) with PEFT LoRA, is kept as the
legacy path. The dataset is plain chat-format JSONL either way, so switching
backends never touches the data tooling.

The trainer is deliberately a swappable backend: nothing outside
`tools/train_local.py` and `tools/train_tinker.py` may assume a specific
backend, so the dataset format, the prompt store, and the benchmark path all
stay trainer-agnostic. Enough unified memory locally also opens up full-rank
fine-tuning of the 4B and a larger ceiling model, neither of which changes
anything upstream of the trainer.

## Design decisions

- Train on the production prompt format verbatim. MinusPod's database stores
  only a placeholder for the detection prompt, so the extractor rebuilds each
  window's prompt with MinusPod's own `create_windows` and
  `format_window_prompt` functions against the stored transcript segments.
  The benchmark harness reconstructs prompts the same way.
- Output schema is the production one: a JSON array of
  `{start, end, confidence, category, reason, end_text}` objects.
- One output span is one contiguous ad break; gaps under 15 seconds merge.
  Enforced in the training labels (`tools/fix_labels.py`), the validator, the
  eval scorer, and the MinusPod benchmark scorer (canonicalized before IoU
  matching). `end_text` is validated at 1-5 words: the prompt asks for 3-5
  but a short span can contain fewer transcript words.
- Every example carries a provenance tier (`human_verified`,
  `machine_accepted`, `hard_negative`, `synthetic`) so training runs can
  ablate by label quality.
- Windows with no ads are kept at their natural ratio (roughly two thirds of
  all windows) rather than downsampled. Whole episodes with zero ads are
  included too. A model that cannot say "no ads here" is useless, since it
  decides what gets cut from the audio.
- The benchmark corpus episodes (`data/holdout.txt`) are eval-only. The
  extractor refuses to emit them and the validator checks again. Some
  extracted feeds overlap benchmark feeds at the show level; the extractor
  prints that overlap so scores are read honestly.
- Validation split is by feed, not by random window, so val measures
  generalization to unseen shows.
- The answer has to be prefix-invariant. Every assistant turn in the data is
  a bare JSON array and the system prompt already forbids prose, but that is
  not enough on its own: served through a template that opens a think block,
  the first checkpoint answered in prose and ran to the token cap, because
  that prompt shape never appeared in training. Serving with
  `--default-chat-template-kwargs '{"enable_thinking": false}'` fixes it, and
  training on both the open and closed prefixes with the same JSON target
  removes the dependence on getting the serving flag right.
- Base model: Qwen, chosen from MinusPod benchmark results. Qwen3.5-9B is now
  the primary training target on local hardware, with Qwen3.5-4B kept as the
  fallback; serving a 9B in the 15 GB budget requires int4/AWQ, validated
  before release. Qwen3.5-4B is also the newest 4B-class dense Qwen; the 3.6
  and 3.8 generations shipped nothing dense under 27B, and Qwen3.8-Flash-Next
  is a 125B-total MoE under a non-Apache license. Inference picks the size,
  not training: a 27B at int4 is 14 to 15 GB of weights alone and leaves no
  room for a 16k KV cache on a 16 GB card. Qwen3.8-27B stays the ceiling
  variant for 24 GB and up. Thinking mode stays off: the task needs fast
  deterministic JSON.
- Local training does not relax the serving budget, but it does lift the
  training ceiling. A full fine-tune of the 4B, rather than LoRA at rank 16,
  is the largest lever available against the same deployment target, and LoRA
  rank plausibly limits how far the span-merging behavior can be retrained.
- Release formats: merged bf16, AWQ, and GGUF q4_K_M for Ollama users.

## Phases

1. Vertical slice (current): minimal extractor, small LoRA run on Tinker,
   serve with vLLM, score with the MinusPod benchmark harness against the
   frozen prompt. Deliverable: a scored benchmark row and a working pipeline.
2. Dataset build-out, planned in detail against a measured inventory of the
   source instance in `phase2-data-plan.md`:
   prefix-invariant training (every example rendered under
   both the thinking-open and thinking-closed assistant prefixes against the
   same JSON target, so no serving flag is load-bearing); full tier
   assignment, hard negatives from rejected
   markers, category backfill for older markers (rules first, then LLM
   few-shot seeded from already-categorized markers, written back through a
   new MinusPod API endpoint), augmentation (community-pattern injection,
   boundary jitter, measured category rebalance, synthetic capped near 25%
   of the mix and never in eval), contributor docs.
3. Real training runs: tier ablations, full 4B run, larger ceiling run,
   pick the release candidate.
4. Release: Hugging Face weights and model card, published benchmark row.

Success criteria for phase 3 exit: F0.5 at or above the current production
model tier on the benchmark corpus, both no-ad control episodes passed, JSON
compliance 1.0, and a few seconds of latency per window on a 16 GB card.

## Status

Done so far:

- Repo scaffolding: example schema, prompt store with hash-deduplicated
  system prompts, holdout list, extraction, validation, dataset build, and
  Tinker training tools. See the README for usage.
- First slice extracted from a MinusPod instance: 17 episodes across 17
  feeds, 200 windows (64% with no ads), 75 ad spans. Validation passes with
  zero errors and no holdout leakage.
- Chat-format train/val split built: 156 train and 44 val examples, val held
  out as three whole feeds.
- Spot checks confirmed window-clipped ads get their `end_text` recomputed at
  the clip boundary, and empty windows serialize as an empty array.
- Dataset migrated to the per-break output contract: outro and audio-only
  spans dropped per the label audit (with a kept-exceptions list at
  `data/keep_spans.json`), sub-15s gaps merged, validator tightened.
- Local trainer landed: `tools/preflight.py` gates training on a matching
  data/model/device stamp, `tools/train_local.py` runs BF16 LoRA against a
  run manifest, `tools/eval_generation.py` scores the held-out val set, and
  `tools/export_local.py` exports the adapter and a merged model behind an
  equivalence gate.

- First LoRA run finished on Tinker: Qwen3.5-4B, rank 16, 3 epochs, 50 steps.
  Final train nll 0.275, test nll 0.282 on the in-run split.
- Merged weights exported and served with vLLM on a 16 GB card. See
  `phase1-runbook.md` for the serving config and the flags it needs.
- Serving verified against real windows: clean JSON arrays, correct stop
  behavior, and 4 to 5 seconds per window, which puts a single-trial
  benchmark run at roughly 15 minutes.

- Phase 1 is complete. The checkpoint scored tier B, F0.5 0.686, with 1.00
  JSON compliance, both no-ad controls passed, 3.6s median per window, at no
  cost per episode. Placed against the roster in [MinusPod's benchmark
  report](https://github.com/ttlequals0/MinusPod/blob/main/benchmarks/llm/results/report.md)
  that ties `gemini-3.1-flash-lite` and clears every untuned open-weight
  entry, against tier A and 0.781 for the hosted model it would replace.
  Full numbers in `runs/20260830-162400-results.md`. This F0.5 0.686 row
  predates scorer canonicalization and is not directly comparable to scores
  produced after it.

Next up:

- Phase 2: more data and longer training. The failure is entirely recall
  (0.738 precision against 0.592 recall), concentrated in short ads, 0.25 at
  under 30 seconds, and post-roll ads at 0.36. The model merges adjacent
  spots into one span, so the other ads in a block become misses.
- Score the untuned base the same way, for a clean before and after.
- Rerun against the frozen 2026-08 prompt snapshot so the row is directly
  comparable to the published table.

Open items and known gaps:

- `audio_context` (audio cue and volume signals MinusPod appends to prompts)
  is omitted from slice examples and flagged in provenance. Phase 2 rebuilds
  it from the stored audio analysis so training matches inference.
- Episodes with markers still pending human review are skipped whole. That
  is the single biggest source of unused episodes in the source instance and
  comes back in phase 2 with finer-grained filtering.
- Markers that predate MinusPod's category field are skipped until the phase
  2 backfill.
- 8 of the 17 training feeds are shows that also appear in the benchmark
  corpus, though never the same episodes. Phase 1 measured the effect:
  episodes whose feed appeared in training averaged F1 0.650 against 0.617
  for feeds never trained on, a 0.033 gap across six episodes per side. That
  is noise, so the score reflects generalization rather than memorized shows.
  Worth re-measuring as the training set grows.
- One window took 901 seconds against a 20 second p99. Cause unknown; watch
  for it on the next run.
