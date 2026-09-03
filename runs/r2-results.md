# Run r2: first local checkpoint

Qwen3.5-9B at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, LoRA rank
16 over 248 linear modules, 43,278,336 trainable parameters. 3 epochs over 153
examples, batch 1 with 16 accumulation steps, 30 optimizer steps in 100
minutes on a DGX Spark. Peak memory 43.1 GiB.

Held-out loss fell every epoch: 0.5632, 0.5615, 0.5426. Mean training loss
0.938.

## Three models, same eval

Scored by `tools/eval_generation.py` on the feed-held-out validation split, 47
windows holding 23 truth spans. Not the benchmark corpus, so these do not
compare to rows in MinusPod's published table.

| Metric | Base | Adapter | Merged |
|---|---|---|---|
| F0.5 | 0.7826 | **0.8403** | 0.7983 |
| Precision | 0.7826 | 0.8333 | 0.7917 |
| Recall | 0.7826 | 0.8696 | 0.8261 |
| TP / FP / FN | 18 / 5 / 5 | 20 / 4 / 3 | 19 / 5 / 4 |
| JSON compliance | 0.9787 | 1.0000 | 1.0000 |
| No-ad false positives | 2 | 3 | 3 |
| Boundary MAE start / end | 10.11s / 9.66s | 8.31s / 7.81s | 6.84s / 8.22s |

## Analysis

**The base model is already strong.** Untuned Qwen3.5-9B scores 0.783 with
97.9% JSON compliance against the production prompt. That is the number this
project has to beat, and it was not measured until now. Phase 1's framing,
that untuned open-weight models fail this task outright, does not hold for a
9B of this generation: it holds for the smaller models in the benchmark
roster.

**The fine-tune's accuracy gain is two spans.** 20 true positives against 18,
three misses against five, out of 23 truth spans. At that sample size a 95%
interval is roughly plus or minus 0.15 and the two overlap heavily. The
correct statement is that the fine-tune is not worse and looks better, not
that it is better.

**The compliance gain is the more defensible one.** 97.9% to 100% is one
window in fifty that production could not parse, against none. The pipeline
has to read the answer to cut anything, so a formatting failure is a total
failure for that window, and it is the one metric where the gap is not
inside the noise.

**Boundary error stayed around 8 seconds.** For a pipeline that cuts audio
that is material: it either clips the show or leaves ad audible. Part of the
cause is likely in the labels rather than the model, since 20 of the 71
training spans exceed 180 seconds against a prompt that says a break runs 60
to 120, an inheritance from MinusPod's older same-sponsor merge rule.

**The no-ad control moved the wrong way**, 2 false positives to 3. Small
numbers, but that is the expensive error and the direction to watch.

## Merging is not equivalent on this architecture

The export equivalence gate failed: 5 of 8 fixtures generated identically,
maximum logit difference 23.5 with a mean of 0.05. Merging in fp32 and casting
to bf16 on save produced identical results, which rules out merge-time
arithmetic as the cause.

The difference is in storage, not arithmetic. A merged model computes
`round_bf16(W + BA*scale) * x`, while the adapter path computes
`W*x + (B(A*x))*scale`, applying the adapter to activations at runtime without
ever rounding it into a weight. Those are different operations, and no
merge-time dtype changes that.

Scoring the merged model settles what the token comparison could not: F0.5
0.7983 against the adapter's 0.8403, one truth span moving from found to
missed and one prediction from correct to spurious. Within noise at 23 spans,
but not the same model.

The adapter is therefore the artifact of record. vLLM serves LoRA adapters
directly, so adapter-only serving is a deployment path rather than a
workaround. A merged model published for this architecture needs its own eval
and should not be described as the same model.

## Benchmark corpus

Scored through MinusPod's harness on the 14 episode corpus, 12 ad-bearing and
2 no-ad controls, one trial at temperature 0, against the frozen
`prompts/2026-08.txt` snapshot. The adapter was served by vLLM 0.28.0 with
`--enable-lora`, not merged.

| Metric | Value |
|---|---|
| F0.5 | 0.735, 95% CI +/-0.114 |
| Precision / Recall | 0.719 / 0.854 |
| F1 | 0.768 |
| TP / FP / FN | 41 / 19 / 6 |
| JSON compliance | 0.9968, 169 of 171 direct arrays |
| No-ad controls | PASS, PASS |
| Latency p50 / p95 | 1.90s / 30.18s |
| Calls | 171 of 171, no errors |

**Band B against the published roster**, whose A floor is 0.776 and B floor
0.730. The harness's own report labels this tier A, which is an artifact of
scoring a single model: tiers are computed against the leader of whatever
roster ran, so a lone model is always its own leader.

**The failure mode inverted.** Phase 1 was precision 0.738 against recall
0.592, and its problem was missing ads. This run is precision 0.719 against
recall 0.854, with 19 false positives against 6 misses. It now finds almost
everything and over-cuts. Since F0.5 weights precision double, over-detection
costs more than the recall gain earns, which is most of why 0.735 is not
higher.

That direction is consistent with the training labels rather than the model:
20 of the 71 training spans exceed 180 seconds against a prompt that says a
break runs 60 to 120, inherited from MinusPod's older same-sponsor merge rule.
A model taught that ad spans are long and inclusive will over-extend and
over-claim.

**Both no-ad controls passed**, which is the reassuring counterpart to the
held-out split, where the tuned model false-positived on 3 of 24 clean
windows.

**JSON compliance is 0.9968, not 1.00.** 169 of 171 responses were direct
arrays, one arrived in a markdown code block, and one as a truncated single
object. The report rounds it to 1.00. All 171 calls used prompt injection
rather than a native JSON mode, because vLLM's `json_object` grammar forces an
object and would fail every array answer, so this row carries the
prompt-inject caveat rather than the native one.

## What this run does not answer

The benchmark's 0.735 and the held-out split's 0.840 are not comparable and
neither supersedes the other. Different corpora, different episodes, different
prompt snapshot. The split says this checkpoint beats its own base on unseen
shows; the corpus says where it sits against the published roster.

What is still missing is the base and merged models on the corpus. Without
those rows there is no corpus-level answer to the question the held-out split
raised, which is whether the fine-tune beats the untuned 9B by more than
noise. That is two more serving runs and is the next measurement worth
spending GPU time on.

Phase 1's 0.686 is also not a fair comparison. That row predates the scorer's
per-break canonicalization, which raised scores across the roster, so the gap
between 0.686 and 0.735 mixes a model change with a scoring change and cannot
be attributed to either.
