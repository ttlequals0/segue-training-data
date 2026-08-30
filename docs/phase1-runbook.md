# Phase 1 runbook

Every step from an extracted dataset to a scored benchmark row, with the
failures already hit and what fixed them.

## 1. Build the dataset

```sh
uv run python tools/extract.py --db .local/minuspod.db --limit 25
uv run python tools/validate.py
uv run python tools/build_dataset.py --val-feeds <feed-a>,<feed-b>,<feed-c>
wc -l .local/train.jsonl .local/val.jsonl
```

`--val-feeds` holds those feeds out of training entirely. Name two or three,
not the whole roster: every feed listed goes to val, so passing all of them
leaves an empty training file and the trainer runs zero steps without
complaining.

## 2. Train on Tinker

```sh
uv sync --extra train
export TINKER_API_KEY=...
uv run python tools/train_tinker.py --list-models
uv run python tools/train_tinker.py --train .local/train.jsonl
```

Check the printed example count before walking away. The final log line gives
the sampler path used for export.

## 3. Export merged weights

```sh
uv run python tools/export_model.py --tinker-path "tinker://<run-id>/sampler_weights/final"
```

Copy the output directory to the GPU host.

## 4. Serve with vLLM

Docker compose, GPU pinned, model directory mounted at `/model`:

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: segue-vllm
    restart: unless-stopped
    ipc: host
    ports:
      - "8123:8000"
    volumes:
      - /path/to/segue-4b-bf16:/model
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    command: >
      --model /model
      --dtype bfloat16
      --max-model-len 16384
      --gpu-memory-utilization 0.90
      --max-num-seqs 64
      --language-model-only
      --host 0.0.0.0
      --port 8000
```

Two things worth knowing before you watch the logs:

- Cold start takes 5 to 8 minutes on a 16 GB card. Weight load, torch.compile,
  and CUDA graph capture all happen before the port opens. Add
  `--enforce-eager` to trade throughput for a 15 second start while iterating.
- `--max-num-seqs 64` is not optional here. The default of 256 exceeds the
  Mamba cache blocks available at this context length and memory fraction, and
  the engine aborts during CUDA graph capture with
  `max_num_seqs (256) exceeds available Mamba cache blocks`.
- The card has to be empty before startup. vLLM reserves its fraction of
  total VRAM up front, so a leftover process holding a few GiB aborts the
  engine with `Free memory on device cuda:0 ... is less than desired GPU
  memory utilization`. Note that the device pinned by `device_ids` appears
  as `cuda:0` inside the container, so check that host GPU with
  `nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv -i <n>`
  and clear stale containers with `docker rm -f segue-vllm` plus
  `docker compose down --remove-orphans`. If the memory belongs to something
  that has to keep running, fit inside what is left instead:
  `--gpu-memory-utilization 0.60 --max-model-len 12288 --max-num-seqs 16`.
  12288 still clears the largest real prompt, which is about 6.5k tokens
  plus a 4096 token answer, but it leaves under a gigabyte of KV cache, so
  drop the harness to `max_concurrent_calls = 1` as well.

Confirm it is up:

```sh
curl -s http://<gpu-host>:8123/v1/models | jq '.data[].id'
```

## 5. Smoke test before benchmarking

A full benchmark run is 171 windows per trial. Send three windows first:

```sh
uv run python tools/smoke_test.py --base-url http://<gpu-host>:8123/v1 --ads-only --n 3
```

It reports response shape, ad count, finish reason, token usage, and latency,
with and without the JSON constraint the harness applies. What to look for:

- `shape=array` in both modes. The harness parses a JSON array. If the
  constrained mode returns an object, the grammar and the fine-tune disagree,
  and the fix is `response_format = "text"` in the benchmark config.
- `finish=stop`, not `finish=length`. A run to the token cap means the model
  never emitted a stop token, which at roughly 30 tokens per second is minutes
  per call and reads as a hang from the harness side.
- Latency in seconds. If one sequential window takes minutes, the harness will
  time out no matter how the concurrency is set.

## 6. Score with the MinusPod benchmark harness

Install the harness dependencies first. The benchmark imports shared modules
from MinusPod's `src/`, which pull in Flask:

```sh
cd <minuspod>/benchmarks/llm
uv add flask
```

Copy `benchmark.local.toml.example` from this repo to
`<minuspod>/benchmarks/llm/benchmark.toml` and fill in the host. That template
carries a one-model roster and settings tuned for a single local GPU. Then:

```sh
export SEGUE_LOCAL_API_KEY=dummy
uv run benchmark run --dry-run     # prints the call count
uv run benchmark run --snapshot prompts/2026-08.txt
```

There is no concurrency flag. Concurrency lives in the config as
`max_concurrent_calls` and `max_concurrent_per_provider`, and retries as
`max_retries` and `timeout_seconds`.

Rapid `openai._base_client: Retrying request` lines mean the OpenAI SDK is
retrying on its own. Set `max_retries = 0` so the real error surfaces instead
of a retry loop, and check the vLLM logs at the same time: if the server is
idle while the client retries, the requests are failing before they arrive; if
the server is busy, the calls are timing out mid-generation.

## Reading the score honestly

The benchmark corpus is eval-only and the extractor enforces that at the
episode level, so no benchmark episode is ever trained on. Feeds are a
different matter. In the current slice, 8 of 17 training feeds are shows that
also appear in the benchmark corpus. The model has seen other episodes of
those shows, including their recurring sponsor reads and their host patterns.

Read the score in two parts: the five benchmark feeds absent from training are
the generalization signal, and the eight shared feeds measure something closer
to same-show recall. Phase 2 should widen the training feeds so this gap
narrows.
