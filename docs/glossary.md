# Glossary

Terms as this project uses them, alphabetically. Where a number appears, it is
this repository's actual value, not a general default. Coverage spans the task
and its data, LoRA training, the tooling and its provenance, and scoring.

**Adapter**: The trained LoRA weights alone, meaningless without the exact
base model they were trained against. Small, a few hundred MB.

**Alpha (`lora_alpha`)**: A scaling factor on the adapter's contribution, 32
here. The effective scale is `alpha / r`, so 2.0.

**BF16 (bfloat16)**: 16-bit floats with the same exponent range as fp32.
Halves memory versus fp32 with none of fp16's overflow problems.

**Boundary MAE and bias**: For matched spans, mean absolute error on start
and end timestamps, and the mean signed error. Bias shows direction: negative
start with positive end means cuts extending past the ad into show content.

**Calibration**: Whether a model's stated `confidence` matches its actual hit
rate. A model claiming 0.95 that is right 60% of the time is overconfident.

**Category**: One of `sponsor`, `cross_promo`, `self_promo`, `interaction`.
Three further values (`intro`, `outro`, `recap`) are legal only when the
prompt carries a SHOW SEGMENTS section, which no current example does.

**Chat template**: The model-specific markup that turns messages into one
token string. Qwen3.5 thinks by default, so this project renders with
`enable_thinking=False` and trains on that exact prefix.

**DAI (dynamic ad insertion)**: Ads stitched in at download time, so the same
episode differs between fetches. Detectable by comparing fetches even when
the transcript contains no promotional words, which is why a few spans exist
whose evidence is audio rather than text.

**Detection buckets**: Detection rate split by ad length (short under 30s,
medium, long) and position (pre-roll, mid-roll, post-roll). Aggregate scores
hide systematic blind spots; phase 1 was 0.25 on short ads against 0.68 on
long ones.

**Empty window**: A window whose target is `[]`. Two thirds of the dataset.
Kept at natural ratio on purpose: a model that cannot say "no ads here" is
unusable, because it decides what gets deleted.

**`end_text`**: The last few spoken words of a span, used as a boundary
anchor so the cut can be snapped to words rather than to a raw timestamp.
Validated at 1 to 5 words; the prompt asks for 3 to 5, but a short span can
contain fewer.

**Epoch**: One pass over the training set. Three here, hence
`epoch: 0.1046` per step, which is 16 divided by 153.

**Equivalence gate**: Export's final check: greedy generations from
base-plus-adapter and from the reloaded merged model must be token-identical
on fixed windows. A mismatch means the merge changed behavior, and the export
is not publishable.

**Example**: A window plus its target: the rendered system and user prompt,
and the JSON array of ad spans that belongs to it. 200 exist here, 153 train
and 47 validation.

**F0.5**: The same, weighting precision twice as heavily. The headline metric
here, because the errors are not symmetric: a false negative leaves an ad in
the episode, while a false positive deletes part of the show.

**F1**: Harmonic mean of precision and recall, weighting them equally.

**Feed-held-out split**: Validation is three whole shows, never individual
windows of a show that also appears in training. Measures generalization to
unseen podcasts rather than to unseen minutes of familiar ones.

**FN (false negative)**: A real ad span that no prediction matched. The ad
stays in the episode.

**FP (false positive)**: A predicted span that matched no real ad. In this
pipeline that means audio was cut out of the show, so it is the expensive
error and the reason scoring weights precision double.

**Gradient accumulation**: Run several micro-batches and sum their gradients
before stepping the optimizer, so a small batch that fits in memory still
behaves like a large one. Batch 1 with accumulation 16 is identical in effect
to batch 2 with accumulation 8, at half the activation cost.

**Gradient checkpointing**: Discard intermediate activations during the
forward pass and recompute them during the backward, trading roughly a third
more compute for a large memory saving. It cut this run's forward from over
80 GiB to 22 GiB. Note that Hugging Face gates it on the model being in
training mode, so it is silently inert on an eval-mode model.

**Gradient clipping**: Rescale gradients whose norm exceeds a threshold, 1.0
here, so one bad batch cannot blow up the weights. `grad_norm` in the logs is
the pre-clip norm.

**Hard negative**: A span the pipeline or a person explicitly ruled not an
ad. The highest-value negative signal available, and absent from the current
dataset; phase 2 adds it.

**Holdout**: Episodes in `data/holdout.txt` that must never be extracted into
training data, because the benchmark scores against them. Enforced twice, by
the extractor and again by the validator.

**IoU (intersection over union)**: Overlap between two time ranges divided by
their union. 0 is disjoint, 1 identical. A prediction counts as matching a
truth span at IoU >= 0.5.

**JSON compliance**: The fraction of responses that parsed as a bare JSON
array of span objects. A model can be accurate and still useless if the
pipeline cannot read its answer.

**Label masking**: Setting label positions to -100 so they contribute no
loss. Here everything except the assistant's JSON answer is masked, so the
model is graded only on its answer, never on reciting the prompt back.

**Learning-rate schedule**: How the rate changes over training. Linear decay
from 1e-4 to zero here.

**LoRA (low-rank adaptation)**: Freezes the base model and trains small
low-rank matrices alongside each target layer. Here that is 43,278,336
trainable parameters against 8.95B frozen, 0.48%.

**Loss (cross-entropy, NLL)**: Average negative log-probability the model
assigns to the correct next token, over tokens that carry loss. Lower is
better; it is not a percentage and has no fixed scale.

**Marker**: MinusPod's own record of an ad in an episode, the upstream source
of a span. A marker that was cut in production is training signal; one still
held for review is not.

**Matching**: Predictions and truth are paired greedily, highest IoU first,
one to one. A prediction can match at most one truth span and vice versa, so a
single prediction covering two real ads scores as one TP and one FN, not two
TPs.

**Merged model**: The adapter folded into the base weights, producing a
standalone model that needs no PEFT at inference. About 17 GB in bf16.

**No-ad control**: Episodes verified to contain no ads at all. Every
prediction on one is a false positive, so it is a direct test of whether the
model can hold its fire.

**Optimizer step**: One weight update, after a full accumulated batch. This
run has 30: 153 examples in batches of 16, three times over.

**Per-break canonicalization**: Merging spans separated by less than 15
seconds, on predictions and ground truth alike, before matching. Mirrors the
detection prompt's own merge rule, so a model is not punished for splitting
one ad break into adjacent spots, or rewarded for the reverse.

**Pinned revision**: The exact base-model commit, here
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Required rather than defaulted,
because an adapter trained against one snapshot and merged against another
corrupts silently.

**Precision**: Of the spans predicted, the fraction that were real ads,
TP / (TP + FP). Low precision means cutting real show content.

**Prefix invariance**: The property that the answer does not depend on which
assistant prefix the server used. Phase 1 answered in prose when served
through a thinking-open template, because it had never seen that shape.

**Preflight**: The gate that must pass before training: it renders every
example, asserts the label masks, checks the split is disjoint and its
manifest current, verifies CUDA and bf16, and runs one real optimizer step on
the longest window. Fails closed by design.

**QLoRA**: LoRA over a 4-bit quantized base. Deliberately not used here: with
128 GB of unified memory there is no reason to accept quantization error.

**Rank (`r`)**: The inner dimension of those matrices, 16 here. Higher rank
means more capacity and more trainable parameters.

**Recall**: Of the real ads, the fraction found, TP / (TP + FN). Low recall
means ads survive in the output.

**Run manifest (`runs/<run-id>/run.json`)**: Everything needed to reproduce a run:
data and prompt hashes, package versions, CUDA and driver and GPU, repository
commits, the resolved LoRA module list, trainable parameter count, seeds, and
checkpoint paths.

**Span**: One contiguous ad break, as `{start, end, confidence, category,
reason, end_text}`. Not one advertisement: two sponsor reads 5 seconds apart
are one span. See per-break canonicalization.

**Split manifest**: Written beside the built dataset: which feeds are
validation, the example ids in each split, and the hash of each output file.
Preflight rejects a manifest that no longer matches the files.

**Stamp (`.local/preflight.json`)**: Preflight's receipt: hashes of the train
and validation files, the model, the revision, the attention implementation,
and the max length. The trainer refuses to start unless a stamp matches the
inputs it is about to use, which is what stops training on data nobody
checked.

**`target_modules="all-linear"`**: Attach adapters to every linear layer
except the output head: 248 modules across 32 layers, spanning the MLP
projections, the full-attention projections, and the linear-attention
projections.

**Tier (provenance)**: How much a label can be trusted:
`human_verified` (a person checked it), `machine_accepted` (cut in production
without a human decision on that window), `hard_negative` (a span a person
rejected, kept as a counterexample), `synthetic` (generated by augmentation).
Recorded per example so a run can ablate by label quality.

**Tier letter (A to G)**: The benchmark's grouping of models that are
statistically tied on the corpus, by a paired test across episodes. Order
within a tier is not meaningful.

**Tier weights**: Per-provenance sampling weights applied to training data
only. Validation is never sampled, or the split would shift whenever a
training knob moved.

**TN (true negative)**: Not counted here, and worth knowing why. Scoring is
over spans, not over every instant of audio, so there is no enumerable set of
"correct non-ads" to count. This is why accuracy is never quoted: with most of
a podcast being content, any model that predicted nothing would score
extremely well on it. The no-ad control episodes serve that purpose instead.

**Token**: The unit the model reads. Examples here run 4,067 to 8,642 tokens.

**TP (true positive)**: A predicted span that matched a real ad span at
IoU >= 0.5 after canonicalization. The model found an ad that was there.

**Unified memory**: One physical pool shared by CPU and GPU, as on a DGX
Spark. Consequential because the "device total" is the machine's whole RAM,
so an uncapped allocation can reach the kernel OOM killer instead of raising
a catchable CUDA error.

**Warmup**: Ramping the learning rate from zero over the first few steps so
early noisy gradients do not wreck the weights. 3% of steps here, which at 30
steps is under one, which is why step 1 logs a learning rate of 0.

**Window**: One chunk of transcript the model judges independently, 600
seconds with 180 seconds of overlap onto the next. Prompts are rendered per
window, so a window is also one training example.
