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
the effort goes into data breadth. Training runs on
[Tinker](https://thinkingmachines.ai/tinker/) with PEFT LoRA; the dataset is
plain chat-format JSONL, so a local trainer can replace Tinker later without
touching the data tooling.

## Design decisions

- Train on the production prompt format verbatim. MinusPod's database stores
  only a placeholder for the detection prompt, so the extractor rebuilds each
  window's prompt with MinusPod's own `create_windows` and
  `format_window_prompt` functions against the stored transcript segments.
  The benchmark harness reconstructs prompts the same way.
- Output schema is the production one: a JSON array of
  `{start, end, confidence, category, reason, end_text}` objects.
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
- Base model: Qwen, chosen from MinusPod benchmark results. Primary target is
  the smallest current-generation dense model Tinker hosts (Qwen3.5-4B at
  time of writing), with a larger variant as a quality ceiling for bigger
  cards. Thinking mode stays off: the task needs fast deterministic JSON.
- Release formats: merged bf16, AWQ, and GGUF q4_K_M for Ollama users.

## Phases

1. Vertical slice (current): minimal extractor, small LoRA run on Tinker,
   serve with vLLM, score with the MinusPod benchmark harness against the
   frozen prompt. Deliverable: a scored benchmark row and a working pipeline.
2. Dataset build-out: full tier assignment, hard negatives from rejected
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

- First LoRA run finished on Tinker: Qwen3.5-4B, rank 16, 3 epochs, 50 steps.
  Final train nll 0.275, test nll 0.282 on the in-run split.
- Merged weights exported and served with vLLM on a 16 GB card. See
  `phase1-runbook.md` for the serving config and the flags it needs.

Next up:

- Score the served checkpoint with the MinusPod benchmark harness, and score
  the untuned base the same way for comparison.

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
  corpus. No benchmark episode is trained on, but the model has seen other
  episodes of those shows. Phase 1 scores split into a five-feed
  generalization signal and an eight-feed same-show signal; phase 2 should
  widen the training feeds.
