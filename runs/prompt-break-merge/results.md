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

Replaced with a content-based rule:

    - MERGING: Emit ONE segment per ad break, not one per advertiser. A break
      often contains several different sponsors read back to back. Span from the
      start of the first pitch to the end of the last one, including the words
      between them. Start a new segment only when actual show content resumes.

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
wrongly cut content, since spanning a break swallows its transitions.

The adapter and the base scored identically here, 44 / 13 / 5 each. The prompt
determines the behavior.

## Not adopted

Recorded here, not shipped. The rule is MinusPod's detection prompt, so
adopting it is a MinusPod change and it shifts every published benchmark row.

Before that happens, the cost side needs its own evaluation. Wrongly cut
content rose from 176s to 309s across 14 episodes, 13 to 22 seconds per
episode, because spanning a break swallows the host chatter between reads.
F0.5 counts that as a win because it scores spans, not audio, and a listener
does not. Someone has to listen to what the wider spans remove and decide
whether clipped show content is an acceptable price for 134 seconds of
recovered ads. Until then this is a measured result, not a recommendation.

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
