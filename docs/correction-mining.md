# Mining human corrections

A spec for the next training experiment. Measured against a database copy
taken 2026-09-03 and the benchmark corpus at MinusPod f45ce8e2, which fixed
two truth files (see "What the false positives are").

## Why

The first local run produced a null result. On the benchmark corpus the
fine-tuned adapter scored F0.5 0.739 and the untuned base scored 0.739
(0.735 and 0.735 before the corpus fix). The adapter genuinely changed the
model, 25 of 171 answers differ, and none of it reached the score.

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
marker is already `was_cut=False`, which the extractor leaves out of the
completion, so the completion already omits it; that is a hard negative the data carries
without saying so.

## What is available

From `pattern_corrections`, current to 2026-09-03. The second count is
restricted to episodes with a retained transcript, which is the constraint
that matters: without `original_segments_json` a window cannot be rebuilt.

| Type | Count | With transcript | What it is |
|---|---:|---:|---|
| `confirm` | 734 | 641 | A span verified as an ad; see the split below |
| `false_positive` | 300 | 218 | A person ruled a span is not an ad |
| `boundary_adjustment` | 83 | 23 | A person moved a span's edges |
| `create` | 8 | 3 | A person authored a span |
| `auto_promotion` | 9 | 0 | Automated, excluded |

505 distinct episodes carry a correction; 423 of those still have a
transcript.

The `confirm` row is mostly not human. 598 of the 734 are written by
`process_episode` when pass 2 corroborates a held marker (`text_snippet`
starts with `auto-approved:`), and 371 of those carry a `corrected_bounds`
trimmed to the pass-2 span. Only 136 confirms are a person's decision, and
5 of those are trimmed.

| Confirm kind | Count | With transcript |
|---|---:|---:|
| Human, bounds unchanged | 131 | 38 |
| Human, trimmed | 5 | 5 |
| Auto-approved, bounds unchanged | 227 | 227 |
| Auto-approved, trimmed | 371 | 371 |

So human boundary supervision on episodes with a transcript is 23 boundary
adjustments, 5 trimmed confirms, and 3 creates, and the extractor can emit
18 of the adjustments and the 5 trims: the 3 creates and 4 of the
adjustments sit on markers with no category, and one adjustment was
overridden by a later confirm at the original bounds. The 371 auto-approved
trims (mean total edge shift 20.3 seconds, median 21.4) are pass-2 output, the same
detector this project is trying to replace, and stay `machine_accepted`.
They are tagged in provenance so they can be ablated on their own. The 83
boundary adjustments shift 39.8 seconds on average (median 10.0).

For scale, the dataset before this change was 17 episodes and 200 windows.

## What the false positives are

The base run's false positives were categorized by hand before writing the
hypothesis, because the corpus truth is human-verified and each one is
either a real over-cut or a truth gap. Two turned out to be truth gaps and
are fixed in MinusPod f45ce8e2: a missing 68 second pre-roll sponsor read,
and a 76 second network promo left rejected although the reviewer rules
count stand-alone promos. Re-scored on the fixed corpus, base and adapter
both move from 0.735 to 0.739 and stay tied. Base: TP 41, FP 17, FN 8,
precision 0.729, recall 0.804. Adapter: TP 42, FP 18, FN 7.

The 17 remaining base false positives:

| Kind | Count | What happened |
|---|---:|---|
| Fragment of a matched break | 7 | Three truth spans predicted as two or three pieces each, no piece reaching IoU 0.5 |
| Boundary miss on a real ad | 8 | One prediction overlapping a truth span, too short or too long for IoU 0.5. Includes the network promo: both models predicted 5.4 of its 75.6 seconds |
| Not an ad | 2 | A host plugging their own show, and a sponsor-name riff on a stretch of degraded transcript |

Fifteen of seventeen are extent errors on real ads, not detections of
non-ads. Joining the fragments alone would give the base TP 44, FP 10, FN 5,
about F0.5 0.83 (derived from the per-break lists, not a run). Both no-ad
control episodes produced zero predictions from both models.

The instance's rejected markers (of 261
demoted markers, 95 have no category, 53 are `sponsor`, 39 `self_promo`, 39
`outro`, 16 `intro`, 13 `interaction`, 4 `cross_promo`, 2 `recap`) can reach
at most the 2 "not an ad" rows, so hard negatives are a secondary arm. And
the model's `reason` strings are partly confabulated (several cite a sponsor
the window does not contain), so they must not be used as supervision or as
evidence of what the model saw.

## Hypothesis

Training on windows a person corrected teaches the model to emit one span
per ad break at the break's full extent, which raises IoU on the benchmark
corpus.

Prediction: on the fixed corpus, base false positives fall below 17 and
misses below 8 in the same run, with the fragment and boundary rows above
accounting for the change. Recall must not fall below 0.80. Hard negatives
are expected to move no more than 2.

If the fragment and boundary rows do not shrink, the corrections do not
carry extent information the model can use, and the next lever is the
prompt's merge rule or the window size, not the training data.

This is falsifiable in a way "add more data" was not.

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
- `hard_negative`: the window contains a span with a `false_positive`
  correction. Marker state alone cannot identify these: rejected markers sit
  in six different (decision, review, source) combinations in the database,
  so the join is by `original_bounds`: the uncut marker at those bounds
  (within 0.5 seconds, the same match MinusPod's reject path uses to clear a
  hold) and any cut marker whose length the rejected span covers by at least
  half (MinusPod's `CORRECTION_MATCH_MIN_COVERAGE` rule). A covered cut
  marker is dropped from the completion and recorded in `dropped_spans` with
  rule `rejected_but_cut`, and `fix_labels.py` will not merge across it.
- Positive corrections keep a marker only when it is the span they name,
  both edges within 0.5 seconds. A human confirm, create, or adjustment
  that covers a marker without being it, or whose span is not cut, or an
  adjustment the recut has not applied yet (the marker still sits at
  `original_bounds`), marks the marker blocking so the windows it touches
  are skipped like pending ones. A cut marker that overlaps a rejected span
  without being covered by it blocks the same way, unless a person ruled on
  that marker, before or after the rejection, or the rejection no longer
  decides any marker it covered. Corrections are read oldest first and the newest
  decision on a marker wins, except that an auto-approval never overrides a
  person; a re-filed decision takes its newest position.
- Uncut markers nobody ruled on are not negatives: pending holds, markers
  the feed's policy kept (`action_applied` `keep`), and confidence-gated
  `REVIEW` ads without a reviewer verdict block their windows unless a
  correction targets them.
- `human_verified` also covers `boundary_adjustment` and trimmed `confirm`
  windows, labelled in provenance so an ablation can isolate them.
- `machine_accepted`: everything else, as today.

Record the correction types per example in provenance so a run can train on
any subset and so the ablation below is possible. Auto-approved confirms are
recorded as `auto_confirm` and `auto_confirm_trimmed` and never change the
tier.

### Result

Extractor 0.2.0 over the same database copy, `--limit 0`: 738 episodes,
8659 windows (80% empty), 1956 windows skipped for an undecided,
uncategorized, or contradicted marker, 23 episodes with every window
skipped. Tiers: `human_verified` 62, `hard_negative` 145,
`machine_accepted` 8452. 24 human corrections on eligible episodes matched
no marker and were ignored. Five cut markers were covered by a span a
person had rejected; they were dropped and recorded under `dropped_spans`
with rule `rejected_but_cut`, 9 entries across 8 windows because windows
overlap.

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

The run is worth keeping if, on the benchmark corpus at f45ce8e2 and
against the base at 0.739:

- False positives fall below 17 with recall at or above 0.80, and the
  benchmark's per-episode paired test against the base row is significant.
  The two must run in one roster for that test to pair them; separate runs
  cannot be compared this way. One fewer false positive is inside the
  +/-0.115 confidence interval and does not count as confirmation.
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
separate. It addresses the 20 training spans that run past 180 seconds. The
fragment row above makes it more relevant than it looked, since those spans
are the only examples of long single breaks the dataset has, but it is a
label change to the machine-accepted majority and belongs in its own run.

Corpus annotation policy is out of scope here but two cases should be
settled in MinusPod: network bumpers ("Here's a show we recommend") and a
host plugging their own show. The first is now counted as an ad; the second
is still scored as a false positive.

## Risks

The corrections are one person's judgment on one instance's shows. A model
trained on them learns that person's boundary preferences, which is an
improvement over learning another model's guesses but is not ground truth.
The benchmark corpus is the check, since it was annotated separately.

The rejected markers do not resemble most of the benchmark's false
positives; only 2 of 17 are detections of non-ads. A null result on the hard
negative arm is expected and says nothing about hard negatives in general.

The extent signal is also one person's preference. Trimmed confirms record
where that person cut, and the benchmark reviewer rules (include the
transition phrase, end at the final URL) may not match. Boundary MAE on the
benchmark is the check.

Human extent supervision is 31 corrections; the trimmed confirms that
looked like the bulk of it are pass-2 output. Expect boundaries to sharpen
only slightly, and do not read a boundary improvement as significant without
the confidence interval.

## Provenance

- Base run: Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a,
  no adapter. Adapter run: r2 (`runs/r2.json`).
- Corpus: MinusPod f45ce8e2. Both runs were re-scored offline from their
  stored `results/raw/calls.jsonl` after the truth fix; nothing was
  re-captured, and `windows_stale` is false on all 171 rows of each run.
- False positive lists: `~/segue-fp-export/fp-base-corrected.json` on the
  training box (17 records), produced by `fp_extract.py` there.
