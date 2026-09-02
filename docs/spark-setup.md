# DGX Spark setup and first run

The DGX Spark is an aarch64 (Grace) host with a Blackwell GPU (sm_121) and
128 GB of unified memory. Two environment paths; preflight decides which
works, in this order.

## Path 1: uv venv with CUDA wheels (preferred)

    git clone <this repo> && cd segue-training-data
    uv sync --extra local
    uv pip install torch --index-url https://download.pytorch.org/whl/cu130

PyPI aarch64 torch wheels are CPU-only; the cu-series index carries the
CUDA aarch64 builds. Then verify:

    uv run python -c "import torch; print(torch.cuda.get_device_name(0),
    torch.cuda.get_device_capability(0), torch.cuda.is_bf16_supported())"

If this fails or preflight's optimizer-step check reports missing sm_121
kernels, use path 2.

## Path 2: NGC PyTorch container (fallback)

Run the current NGC PyTorch release for DGX Spark with the repo mounted:

    docker run --gpus all -it --rm -v $PWD:/work -w /work \
        nvcr.io/nvidia/pytorch:<current-release>-py3
    pip install transformers==5.5.4 peft accelerate jsonschema

## The pinned revision

Every command below takes `--revision`. It is required, never defaulted: an
adapter trained against one base snapshot and merged against another is a
silent corruption, so the revision is part of the run's identity and is
recorded in `runs/<run-id>.json`.

The pin for Qwen3.5-9B is:

    export REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a

That is the commit `Qwen/Qwen3.5-9B` resolved to when this work was pinned.
Use it verbatim to reproduce these runs. To see what the hub currently
serves, which is not necessarily the pin:

    git ls-remote https://huggingface.co/Qwen/Qwen3.5-9B main

To confirm which snapshot a machine actually has on disk, the directory name
under the cache is the revision:

    ls $HF_HOME/hub/models--Qwen--Qwen3.5-9B/snapshots/

Repinning to a newer snapshot is a deliberate change: update this file, and
treat existing adapters as bound to the revision their manifest records.

## First run

    export HF_HOME=/path/with/space
    hf download Qwen/Qwen3.5-9B --revision $REV

    uv run python tools/build_dataset.py \
      --val-feeds on-air-with-dan-and-alex2,security-now-audio,the-brilliant-idiots \
      --out-dir .local

Build the dataset first. Those three feeds are the held-out validation split
and are held out as whole shows, so val measures generalization to unseen
podcasts rather than unseen windows of familiar ones. Expect 153 train and 47
val examples. Changing the split changes what val means, so keep it fixed
across runs you intend to compare.

    uv run python tools/preflight.py --revision $REV
    uv run python tools/train_local.py --run-id r1 --revision $REV \
        --batch-size 1 --grad-accum 16
    uv run python tools/eval_generation.py --run-id r1 --revision $REV \
        --adapter .local/runs/r1/adapter
    uv run python tools/export_local.py --run-id r1 --revision $REV \
        --adapter .local/runs/r1/adapter

Batch size 1 with 16 accumulation steps is the measured-safe setting on a
121 GiB Spark: preflight peaks at 43.1 GiB on a single 8,642-token window, so
two per micro-batch would crowd the cap. It is the same effective batch and
the same 30 optimizer steps as the batch-2 default.

Preflight must pass before training; the trainer refuses a stale or missing
stamp. The run manifest lands in runs/<run-id>.json and is committed.

## Sizing and the unified-memory trap

Unified memory on GB10 is host RAM. There is no separate VRAM pool: torch
reports the device total as the machine's whole 121 GiB, and an allocation
that targets it competes with the OS, containers, and every other process.
An uncapped run does not raise a catchable CUDA error, it reaches the kernel
OOM killer, which on a first attempt here killed eleven processes including a
container's node process and the driving session.

Both preflight and the trainer therefore cap the allocator with
`--memory-fraction` (default 0.8, about 97 GiB, leaving roughly 24 GiB for
everything else). Lower it on a busier box. The cap converts a machine-killer
into a normal failed check, so do not raise it to 1.0 to make a run fit.

9B BF16 weights are 16.7 GiB. The other large term is the vocabulary: at
248,320 tokens, one 8,642-token window's logits are 4.0 GiB in bf16, 8.0 GiB
upcast for the loss, plus an equal gradient. Preflight prints staged
high-water marks (model loaded, after forward, after backward, after step) so
a run that dies still leaves a measurement behind.

Note that 24 of the 32 layers are linear-attention. Without
`flash-linear-attention` and `causal-conv1d` installed they run an unfused
torch fallback, which the warning at load time announces and which is a
suspect for large activation cost.
