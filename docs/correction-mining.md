# Mining human corrections

A spec for the next training experiment. Measured against a database copy
taken 2026-09-03.

## Why

The first local run produced a null result. On the benchmark corpus the
fine-tuned adapter scored F0.5 0.735 and the untuned base scored 0.735. The
adapter genuinely changed the model, 25 of 171 answers differ, and none of it
reached the score.

The dataset explains that. All 200 training examples carry tier
`machine_accepted`, none reviewed, none corrected. They are production output
from a cloud LLM answering the same prompt. We fine-tuned a 9B on another
model's answers to a task the 9B already performs at the same level, and it
agreed with those answers most of the time already. Volume does not fix that:
more machine-accepted examples teach more agreement.

Meanwhile the instance holds 1,125 human corrections that the extractor
discards by construction. `usable_markers` keeps a marker only when
`was_cut` is true and `held_for_review` is false, and drops the entire episode
if any marker lacks a category. Every place a person overruled the pipeline is
filtered out.

## What is available

From `pattern_corrections`, current to 2026-09-03:

| Type | Count | What it is |
|---|---:|---|
| `confirm` | 734 | A person verified a span is an ad |
| `false_positive` | 300 | A person ruled a span is not an ad |
| `boundary_adjustment` | 83 | A person moved a span's edges |
| `create` | 8 | A person authored a span |
| `auto_promotion` | 9 | Automated, excluded |

505 distinct episodes carry a correction. 423 of those still have a retained
transcript, which is the constraint that matters: without
`original_segments_json` a window cannot be rebuilt.

All 83 boundary adjustments store both `original_bounds` and
`corrected_bounds`. That is a before and after pair rather than a bare label,
and it targets the metric this project is weakest on. Measured boundary error
on the first run was around 8 seconds at each edge; corrections in the
instance average a 14.69 second shift.

For scale, the current dataset is 17 episodes and 200 windows.

## Hypothesis

Training on windows a person corrected reduces false positives on the
benchmark corpus.

Prediction: false positives fall below 18 while recall holds at or above
0.84. Both models currently over-cut, 18 to 19 false positives against 6 to 7
misses, and since F0.5 weights precision double that is what holds both scores
at 0.735.

If false positives do not fall, the ceiling is not the model. The next step
then is auditing the 18 benchmark false positives by hand to find how many are
real ads the truth files never marked, which would mean we are scoring against
a noisy target and no training run can fix it.

This is falsifiable in a way "add more data" was not.

## Design

### Extraction

The corrections do not need a new example format. A window whose span a person
corrected, paired with the corrected answer, is already the training signal:
the pipeline got it wrong, the person fixed it, and the model learns the fix.
What has to change is that these episodes stop being discarded.

1. Filter per marker rather than per episode. Today one pending or
   uncategorized marker drops the whole episode. Instead keep cut markers,
   exclude pending ones from the completion, and keep the rest of the episode.
2. Join `pattern_corrections` on `episode_id` during extraction and record,
   per window, which correction types touched it.
3. Verify before building anything that `ad_markers_json` reflects the
   post-correction state. The recut flow suggests it does. If it does not,
   corrected bounds have to be applied from `corrected_bounds` directly, which
   is a larger change and should be settled first.

### Tiers

Provenance already has the vocabulary; nothing new is needed.

- `human_verified`: the window contains a span with a `confirm` or `create`
  correction.
- `hard_negative`: the window contains a span a person marked
  `false_positive`, and the completion correctly omits it.
- `human_verified` also covers `boundary_adjustment` windows, with the
  adjustment recorded in provenance so an ablation can isolate them.
- `machine_accepted`: everything else, as today.

Record the correction types per example in provenance so a run can train on
any subset and so the ablation below is possible.

### Holdout

Unchanged and non-negotiable. Benchmark corpus episodes are excluded by the
extractor and checked again by the validator. Corrections on a holdout episode
are dropped with the episode. Expect the correction set to overlap the
benchmark shows at the feed level, which is already true of the current data
and already reported.

### Split

Validation stays feed-held-out. Adding 423 episodes changes the split's
composition, so the val feeds should be reselected and then frozen, and the
old and new splits must not be compared as though they measured the same
thing.

## Success criteria

The run is worth keeping if, on the benchmark corpus and against the base at
0.735:

- False positives fall below 18 with recall at or above 0.84.
- Both no-ad controls still pass.
- JSON compliance stays at or above 0.9968.

Report it against the base row, not against the first fine-tune, since the
first fine-tune and the base are the same number.

## Ablation

Run at least two configurations, or the result cannot be attributed:

1. Everything: machine-accepted plus human-corrected.
2. Human-corrected only.

If the second matches or beats the first, the machine-accepted majority is
inert and the dataset should shrink rather than grow, which would be a useful
and cheap finding.

## Out of scope

Category backfill for the 666 uncategorized markers, `audio_context`
restoration, re-windowing for multi-ad coverage, and synthetic augmentation.
Those are the phase 2 plan's items and none of them test this hypothesis.

Re-extraction under MinusPod's corrected same-sponsor merge rule is also
separate. It is worth doing, and it addresses the 20 training spans that run
past 180 seconds, but the base row showed over-detection is not caused by
those labels.

## Risks

The corrections are one person's judgment on one instance's shows. A model
trained on them learns that person's boundary preferences, which is an
improvement over learning another model's guesses but is not ground truth.
The benchmark corpus is the check, since it was annotated separately.

83 boundary adjustments is a small set. Expect it to sharpen boundaries only
slightly, and do not read a boundary improvement as significant without the
confidence interval.
