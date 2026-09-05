# Run ablation-b: human-corrected windows only

Qwen3.5-9B at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, LoRA rank
16, 43,278,336 trainable parameters. 3 epochs over 205 examples, batch 1 with
16 accumulation steps, 39 optimizer steps in 7 hours 37 minutes on a DGX
Spark. Peak memory 42.0 GiB.

The arm trains only on windows a person ruled on: `human_verified` and
`hard_negative`, reached with `--tier-weights machine_accepted=0`. It exists
to test whether corrections carry signal the pipeline's own output does not.

Held-out loss fell every checkpoint: 0.2866, 0.2748, 0.2537. Mean training
loss 0.9153. The validation split is 76.9% empty completions against 54.6% in
training, so the absolute loss is flattered and only its direction is
informative.

## Held-out split

`tools/eval_generation.py` over the same 104 windows for both models. Not the
benchmark corpus.

| Metric | Base | Adapter |
|---|---|---|
| F0.5 | 0.7563 | 0.7944 |
| Precision | 0.7826 | 0.8500 |
| Recall | 0.6667 | 0.6296 |
| TP / FP / FN | 18 / 5 / 9 | 17 / 3 / 10 |
| Boundary MAE start / end | 14.96s / 12.21s | 11.28s / 18.19s |

Two fewer false positives against one more miss, on 27 truth spans. That is
inside noise, and it did not survive the corpus.

## Benchmark corpus

14 episodes, both models served from one vLLM container and scored in one
roster, so the paired per-episode test applies.

| Metric | Base | Adapter |
|---|---|---|
| F0.5 | 0.754 | 0.739 |
| TP / FP / FN | 42 / 18 / 7 | 42 / 18 / 7 |
| Boundary MAE start / end | 4.67s / 7.54s | 5.31s / 6.33s |

Identical detection counts. Both land in tier A, meaning the paired test found
no consistent difference, so the F0.5 gap is not a ranking.

## What it settles

Run r2 trained on 8555 `machine_accepted` windows and matched the base. This
arm trained on 205 human-corrected windows and also matched the base, within
0.2s of r2 on both boundary metrics. Two datasets 40 times apart in size, one
machine and one human, produce the same model.

So the boundary shift r2 showed is a generic effect of LoRA on this task, not
signal from corrections, and there is no measured advantage to human-corrected
data. The model saturates immediately here, which also rules out dataset size
as the explanation for r2's null result.
