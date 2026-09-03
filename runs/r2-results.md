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

## What this run does not answer

The validation split cannot separate 0.783, 0.798, and 0.840. Distinguishing
them needs the benchmark corpus, which is the same corpus the published table
uses and is large enough for the paired test the tiering depends on.
