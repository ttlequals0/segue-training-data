# Mining human corrections

A spec for the next training experiment. Measured against a database copy
taken 2026-09-03.

## Why

The first local run produced a null result. On the benchmark corpus the
fine-tuned adapter scored F0.5 0.735 and the untuned base scored 0.735. The
adapter genuinely changed the model, 25 of 171 answers differ, and none of it
reached the score.

The dataset explains that. All 200 training examples carry tier
`machine_accepted` with `reviewed` and `corrected` false. That is the tag, not
the data: 10 of the 17 episodes carry corrections (8 confirms across 6
episodes, 5 false positives across 4), and the extractor never looks at
`pattern_corrections`, so it cannot tell them apart. The other 7 episodes
are production output from a cloud LLM answering the same prompt. We
fine-tuned a 9B on another model's answers to a task the 9B already performs
at the same level, and it agreed with those answers most of the time already.
Volume does not fix that: more machine-accepted examples teach more agreement.

Meanwhile the instance holds 1,125 human corrections, and the extractor uses
almost none of them. The dataset is 17 episodes because `extract.py --limit`
defaults to 25 and round-robin exhausts small feeds, not because corrections
are filtered. What does block correction episodes: `pending_review_count > 0`
drops the whole episode (225 of the 423 with transcripts), and one
uncategorized cut marker drops the whole episode (108 of 423). 129 correction
episodes pass both filters today and are not selected. A rejected
marker is already `was_cut=False`, which `usable_markers` skips per marker, so
the completion already omits it; that is a hard negative the data carries
without saying so.

## What is available

From `pattern_corrections`, current to 2026-09-03. The second count is
restricted to episodes with a retained transcript, which is the constraint
that matters: without `original_segments_json` a window cannot be rebuilt.

| Type | Count | With transcript | What it is |
|---|---:|---:|---|
| `confirm` | 734 | 641 | A person verified a span is an ad |
| `false_positive` | 300 | 218 | A person ruled a span is not an ad |
| `boundary_adjustment` | 83 | 23 | A person moved a span's edges |
| `create` | 8 | 3 | A person authored a span |
| `auto_promotion` | 9 | 0 | Automated, excluded |

505 distinct episodes carry a correction; 423 of those still have a
transcript.

Boundary supervision is larger than the `boundary_adjustment` row suggests.
376 of the confirms are trimmed confirms whose `corrected_bounds` differ from
`original_bounds`, a mean total edge shift of 21.8 seconds (median 21.6).
The 83 boundary adjustments shift 39.8 seconds on average (median 10.0), but
only 23 of them sit on episodes with a transcript. Both are before and after
pairs rather than bare labels, and they target the metric this project is
weakest on: measured boundary error on the first run was around 8 seconds at
each edge.

For scale, the current dataset is 17 episodes and 200 windows.

## Hypothesis

Training on windows a person corrected reduces false positives on the
benchmark corpus.

Prediction: false positives fall below 18 while recall holds at or above
0.84. Both models currently over-cut, 18 to 19 false positives against 6 to 7
misses, and since F0.5 weights precision double that is what holds both scores
at 0.735.

If false positives do not fall, the ceiling is not the model, and the next
question is what the model is over-cutting and whether these corrections say
anything about it.

This is falsifiable in a way "add more data" was not.

## Before building

Categorize the 18 benchmark false positives by hand first. The corpus truth
files are human-verified, so these are real over-cuts, and sorting them
(sponsor read, self promo, cross promo, intro or outro, host interaction)
costs nothing. It decides whether the instance's corrections can reach them.
The human-rejected markers on the instance skew toward non-sponsor content:
of 261 demoted markers, 95 have no category, 53 are `sponsor`, 39
`self_promo`, 39 `outro`, 16 `intro`, 13 `interaction`, 4 `cross_promo`,
2 `recap`. If the 18 are mostly sponsor reads, hard negatives about intros and
outros will not move them.

## Design

### Extraction

The corrections do not need a new example format. A window whose span a person
corrected, paired with the corrected answer, is already the training signal:
the pipeline got it wrong, the person fixed it, and the model learns the fix.
What has to change is that these episodes stop being discarded.

1. Filter per window rather than per episode. Today one pending or
   uncategorized marker drops the whole episode. Instead skip only the
   windows that intersect a pending or uncategorized marker and keep the
   rest. Do not emit those windows with the marker removed: a held marker is
   undecided and an uncategorized cut marker is a real ad, so a completion
   that omits either teaches "no ad here" for a span that may be one.
2. Join `pattern_corrections` by bounds, not by `episode_id` alone.
   Corrections carry no window index, so intersect `original_bounds` and
   `corrected_bounds` with the window range the way `window_completion`
   clips markers, and record per window which correction types touched it.
3. `ad_markers_json` already reflects the post-correction state, checked
   against the database copy by matching each correction's bounds to the
   current marker within 0.5 seconds:

   | Correction | Marker now | Count |
   |---|---|---:|
   | `false_positive` | `was_cut=False`, `validation.decision=REJECT` | 191 |
   | `false_positive` | no marker at those bounds | 24 |
   | `false_positive` | still `was_cut=True` | 3 |
   | `confirm` (trimmed) | cut at `corrected_bounds` | 373 |
   | `confirm` | cut at `original_bounds` | 255 |
   | `confirm` | no marker at those bounds | 13 |
   | `boundary_adjustment` | cut at `corrected_bounds` | 22 |

   No separate application of `corrected_bounds` is needed. The 3 still-cut
   rejections come from rejecting a marker that was already cut: the reject
   path only demotes pending markers. Drop those spans from the completion.
   The 37 no-marker rows are corrections on episodes reprocessed since; tag
   them stale and do not use them to label a window.

### Tiers

Provenance already has the vocabulary; nothing new is needed.

- `human_verified`: the window contains a span with a `confirm` or `create`
  correction.
- `hard_negative`: the window contains a marker with `was_cut=False` and
  `validation.decision=REJECT`, so the completion omits a span the pipeline
  detected and a person rejected.
- `human_verified` also covers `boundary_adjustment` and trimmed `confirm`
  windows, with the shift recorded in provenance so an ablation can isolate
  them.
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

- False positives fall below 18 with recall at or above 0.84, and the
  benchmark's per-episode paired test against the base row is significant.
  The two must run in one roster for that test to pair them; separate runs
  cannot be compared this way. One fewer false positive is inside the
  +/-0.114 confidence interval and does not count as confirmation.
- Both no-ad controls still pass.
- JSON compliance stays at or above 0.9968.

Report it against the base row, not against the first fine-tune, since the
first fine-tune and the base are the same number.

## Ablation

Run at least two configurations, or the result cannot be attributed:

1. Everything: machine-accepted plus human-corrected.
2. Human-corrected only: windows tagged `human_verified` or `hard_negative`.
   Most windows in a corrected episode touch no correction, so this is a
   window-level subset, not an episode-level one.

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

The rejected markers may not resemble the benchmark's false positives. The
categorization above is the check; if the kinds do not overlap, a null result
says nothing about hard negatives in general.

23 usable boundary adjustments is a small set; the 376 trimmed confirms carry
most of the boundary signal. Expect boundaries to sharpen only slightly, and
do not read a boundary improvement as significant without the confidence
interval.
