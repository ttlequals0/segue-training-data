# Prompt change: one segment per ad break

No training. A one-line edit to MinusPod's detection prompt, scored on the
same 14-episode corpus and the same vLLM server as `ablation-b`.

## Why

Scoring the corpus in audio seconds rather than span counts moved the problem.
Of 5222 seconds of ad audio, the base model caught 86.7% and wrongly cut 176
seconds. The 696 seconds it missed decompose as:

| Where the missed audio sits | Seconds |
|---|---|
| Ads never detected (2 of 53) | 50 |
| Late starts on ads it found | 177 |
| Early ends on ads it found | 201 |
| Holes between reads inside one break | 268 |

Detection was never the problem. The model finds 51 of 53 ads, then emits one
span per advertiser and drops the transitions between them. That loses ad
audio and turns one break into several spans, and the extra spans are charged
as false positives. One behavior produced both halves of the score.

## The change

The prompt's merge rule was numeric and the model did not honor it across real
gaps:

    - MERGING: Multiple ads with gaps < 15 seconds = ONE segment

Replaced with a content-based rule: emit one segment per break, spanning the
first pitch to the last including the words between them, and start a new
segment only when show content resumes.

## Result

| Metric | Original prompt | Break-level prompt |
|---|---|---|
| F0.5 | 0.754 | 0.798 |
| TP / FP / FN | 42 / 18 / 7 | 44 / 13 / 5 |
| Ad audio caught | 86.7% | 89.2% |
| Ads left in | 696s | 562s |
| Content wrongly cut | 176s | 309s |
| Holes inside breaks | 268s | 166s |

Both halves moved at once. Recovering 134 seconds of ads costs 133 seconds of
wrongly cut content, since spanning a break swallows its transitions. F0.5
prices that as a clear win; whether it is the right trade depends on how much
a few seconds of clipped show content matters against leaving ads in.

The adapter and the base scored identically here, 44 / 13 / 5 each. The prompt
determines the behavior.

## Where it belongs

The rule is MinusPod's detection prompt, not this repo. Adopting it is a
MinusPod change, and it shifts every published benchmark row.

## Also checked

MinusPod's validator merges ads across gaps up to `MAX_SILENT_GAP` (30s) when
no speech sits between them, wider than the benchmark's 15s canonical gap.
Rescoring predictions through that rule changes nothing: the gaps inside a
break are full of ad speech, so the validator would not merge them either. The
split spans are real and would reach the cutting step.

Sampled ad markers on a production instance show consecutive markers are
almost always in different breaks: 88% are more than 300 seconds apart, and
1.0% of 576 gaps fall in the 15 to 30 second band. Widening the benchmark's
canonical gap would not model how ads are actually laid out.
