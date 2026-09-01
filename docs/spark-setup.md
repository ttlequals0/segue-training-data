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

## First run

    export HF_HOME=/path/with/space
    hf download Qwen/Qwen3.5-9B --revision <pinned-sha>

    uv run python tools/preflight.py --revision <pinned-sha>
    uv run python tools/train_local.py --run-id r1 --revision <pinned-sha>
    uv run python tools/eval_generation.py --run-id r1 --revision <pinned-sha> \
        --adapter .local/runs/r1/adapter
    uv run python tools/export_local.py --run-id r1 --revision <pinned-sha> \
        --adapter .local/runs/r1/adapter

Preflight must pass before training; the trainer refuses a stale or missing
stamp. The run manifest lands in runs/<run-id>.json and is committed.

## Sizing

9B BF16 weights are ~18 GB; LoRA training with gradient checkpointing and
16k-token windows fits comfortably in unified memory. Preflight's optimizer
step prints measured peak memory for the longest window.
