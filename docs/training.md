# Training Segue yourself

This is the whole path from a fresh machine to a trained, evaluated, exported
LoRA adapter. Every command is meant to be run verbatim. Where a number
appears as expected output, it was measured on a DGX Spark unless the text
says otherwise.

Terms used here without explanation are defined in
[glossary.md](glossary.md).

## What you get

A LoRA adapter over Qwen3.5-9B that reads a rendered podcast transcript window
and answers with a JSON array of ad spans, in the exact format MinusPod's
pipeline parses. The pipeline is four gated steps: preflight, train, evaluate,
export. Each refuses to run on inputs the previous one did not bless.

## Hardware you need

The measured peak is 43.1 GiB on the longest window in the dataset, 8,642
tokens. Plan for a single CUDA GPU with 48 GB or more: an A6000, L40S, A100,
H100, or a DGX Spark. A 24 GB card will not fit this configuration.

Two Spark-specific facts, because they cost a day here:

Unified memory is host RAM. There is no separate VRAM pool. Torch reports the
device total as the machine's entire 121 GiB, so an allocation that targets it
competes with the OS, your containers, and your shell. An uncapped run does
not raise a catchable CUDA error, it reaches the kernel OOM killer. On the
first attempt here that killed eleven processes, including a container's node
process and the session driving the work. Both preflight and the trainer
therefore cap the allocator with `--memory-fraction`, default 0.8. Lower it on
a busy box. Do not raise it to 1.0 to make a run fit.

The Spark is aarch64. PyPI's aarch64 torch wheels vary in whether they carry
CUDA, so verify rather than assume (see Setup, step 3).

You also need roughly 25 GB of disk for the model cache, and a CUDA driver new
enough for your torch build.

## Setup

1. Install [uv](https://docs.astral.sh/uv/), then clone and sync:

       git clone https://github.com/ttlequals0/segue-training-data.git
       cd segue-training-data
       uv sync --extra local

2. Give the venv a Python that ships its own headers. Triton JIT-compiles a
   small C file on the first GPU kernel launch, so it needs `Python.h`. A venv
   built on a system Python without its `-dev` package fails at the first
   kernel with `fatal error: Python.h: No such file or directory`. The
   uv-managed interpreter includes headers and needs no root:

       uv python install 3.12
       uv venv --python 3.12 && uv sync --extra local

   Installing your distribution's `python3.12-dev` works too, if you have root
   and prefer the system interpreter.

3. Verify torch actually reaches the GPU:

       uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.is_bf16_supported())"

   You want `True`, your device name, its compute capability, and `True` for
   bf16. On a Spark that reads `2.13.0+cu130 True NVIDIA GB10 (12, 1) True`.
   If CUDA is unavailable, install a CUDA build for your platform, for example
   `uv pip install torch --index-url https://download.pytorch.org/whl/cu130`,
   and re-run the check.

## The pinned revision

Every command takes `--revision`, and it is required rather than defaulted. An
adapter trained against one base snapshot and merged against another is a
silent corruption, so the revision is part of a run's identity and is recorded
in `runs/<run-id>.json`.

    export REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a

That is the commit `Qwen/Qwen3.5-9B` resolved to when this work was pinned.
Use it verbatim to reproduce these runs. To see what the hub serves now, which
is not necessarily the pin:

    git ls-remote https://huggingface.co/Qwen/Qwen3.5-9B main

To confirm what a machine already has, the snapshot directory name is the
revision:

    ls $HF_HOME/hub/models--Qwen--Qwen3.5-9B/snapshots/

Repinning is a deliberate change: update this file, and treat existing
adapters as bound to the revision their manifest records.

## Download the base model

    export HF_HOME=/path/with/25GB/free
    uv run hf download Qwen/Qwen3.5-9B --revision $REV

`HF_HOME` is not persisted anywhere. Export it in every shell that runs these
commands, or the tools will re-download.

## Build the dataset

    uv run python tools/build_dataset.py \
      --val-feeds on-air-with-dan-and-alex2,security-now-audio,the-brilliant-idiots \
      --out-dir .local

Expect `train: 153  val: 47  downsampled: 0`, plus a `split_manifest.json`
recording the split and the file hashes.

Those three feeds are held out as whole shows, so validation measures
generalization to unseen podcasts rather than unseen windows of shows the
model already trained on. Changing the split changes what val means, so keep
it fixed across runs you intend to compare.

## Preflight

    uv run python tools/preflight.py --revision $REV

This is the gate. It renders all 200 examples through the exact chat template
and asserts the label masks, checks the split is disjoint and the manifest is
current, verifies CUDA and bf16, and then runs one real forward, backward, and
optimizer step on the longest window. A passing run looks like this:

    PASS train file: 153 rows
    PASS val file: 47 rows
    PASS split disjoint
    PASS render train: {'n': 153, 'tokens_min': 4067, 'tokens_median': 6930, 'tokens_max': 8642, ...}
    PASS render val: {'n': 47, ...}
    INFO expected optimizer steps: 30
    PASS cuda visible: NVIDIA GB10
    PASS bf16 supported
    INFO attention implementation: sdpa
      gradient checkpointing active: True (training=True)
      model loaded: peak 16.8 GiB
      after forward: peak 39.2 GiB
      after backward: peak 43.1 GiB
      after optimizer step: peak 43.1 GiB
    PASS optimizer step: {'loss': 1.744675874710083, 'peak_memory_gb': 43.1, ...}
    preflight PASSED; stamp written to .local/preflight.json

Those staged high-water marks are the point: a run that dies still leaves a
measurement behind, and the gap between them tells you where the memory went.

Preflight fails closed on purpose. If a check fails, fix the cause. Do not
pass `--skip-model-step` to get a green run: that flag exists for machines
without a GPU, and using it to dodge a failure defeats the gate.

## Train

    uv run python tools/train_local.py --run-id r1 --revision $REV \
      --batch-size 1 --grad-accum 16

Batch size 1 with 16 accumulation steps is the measured-safe setting: one
8,642-token window peaks at 43.1 GiB, so two per micro-batch would crowd the
cap. It is the same effective batch and the same 30 optimizer steps as the
batch-2 default.

The trainer requires a fresh preflight stamp matching the same data hashes,
model, revision, and max length, so run it in the same shell with the same
exports. Watch for `trainable params: 43,278,336 (expected 43,278,336)`. A
mismatch aborts by design: it means PEFT targeted a different module set than
this configuration expects, which is worth understanding rather than
overriding.

Evaluation loss against the held-out feeds is logged every 10 steps, the
adapter lands in `.local/runs/r1/adapter`, and `runs/r1.json` records the full
run manifest: data and prompt hashes, package versions, driver and GPU,
repository commits, LoRA module list, and seeds. Commit that manifest.

## Evaluate

    uv run python tools/eval_generation.py --run-id r1 --revision $REV \
      --adapter .local/runs/r1/adapter

Greedy decoding over the held-out feeds, scored the way MinusPod's benchmark
scores: spans on both sides are merged into contiguous breaks (gaps under 15
seconds) before IoU matching at 0.5. Reports JSON compliance, precision,
recall, F0.5, false positives on no-ad windows, and boundary error. Results
land in `.local/eval-r1.json`.

## Export

    uv run python tools/export_local.py --run-id r1 --revision $REV \
      --adapter .local/runs/r1/adapter

Writes the PEFT adapter and a merged BF16 model, checksums both, then gates on
equivalence: greedy generations from base-plus-adapter and from the reloaded
merged model must be token-identical on held-out fixtures. A mismatch exits
non-zero and the export is not publishable. BF16 merge rounding can
legitimately flip a token; if that happens, investigate the precision rather
than forcing past the gate.

## Troubleshooting

Every entry here is a failure that actually happened during bring-up.

**`fatal error: Python.h: No such file or directory`** during the first GPU
kernel. Triton cannot compile its helper. Rebuild the venv on a uv-managed
Python, or install your system `python3.N-dev` package. See Setup, step 2.

**`torch.cuda.is_available()` hangs forever**, no error, no traceback. A CUDA
MPS control daemon is running and its handshake never completes. Check with
`ps aux | grep nvidia-cuda-mps`. If the daemon and its clients are stale, shut
it down with `echo quit | nvidia-cuda-mps-control`. If it is serving a live
workload that you must not disturb, bypass it for your process only with
`export CUDA_MPS_PIPE_DIRECTORY=/nonexistent`.

**The machine's userspace dies mid-run and processes get SIGKILLed.** On
unified-memory hardware you allocated past what the host can spare and reached
the kernel OOM killer. Lower `--memory-fraction`. If you removed the cap,
restore it.

**`undefined symbol: _ZN3c104cuda...` on import.** A compiled extension such
as `causal-conv1d` was built against a different torch than the one you run.
Either uninstall it or rebuild against your torch with `uv pip install
--no-build-isolation <package>`.

**`The fast path is not available ... falling back to torch implementation`.**
24 of the 32 layers are linear-attention and run an unfused path without
`flash-linear-attention` and `causal-conv1d`. Measured here, this costs speed,
not memory: the backward added only 3.9 GiB. Both packages are optional, and
`causal-conv1d` compiles from source, so treat them as a speed experiment
rather than a requirement.

**Preflight fails on the trainable-parameter assertion.** PEFT matched a
different set of linear modules than the expected 248. Report the printed
module list rather than relaxing the assertion; a silent mismatch means you
are training something other than what the manifest claims.

## Reporting a run

A run is only comparable if its provenance is. Keep `runs/<run-id>.json`, the
eval JSON, and the export manifest together, and state the base revision, the
val feeds, and the memory fraction alongside any score you publish.
