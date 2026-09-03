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

This section describes the phase 1 setup: a merged 4B on a 16 GB card with two
GPUs present. For the current checkpoint see `training.md`, and note three
differences measured on a DGX Spark. Serve the adapter with `--enable-lora`
rather than a merged model, since merging is not equivalent on this
architecture. Set `--gpu-memory-utilization` near 0.45 rather than 0.90,
because unified memory is host RAM and 0.90 of it reaches the kernel OOM
killer. And a single-GPU host has no device to pin, so the `device_ids` entry
below does not apply.

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
      --default-chat-template-kwargs '{"enable_thinking": false}'
      --host 0.0.0.0
      --port 8000
```

`--default-chat-template-kwargs` is the flag that makes serving match
training, and without it the model is unusable. Tinker's
`qwen3_5_disable_thinking` renderer closes the think block before the answer,
which is what teaches the model to open with a JSON array. vLLM applies the
base Qwen3.5 template, which leaves the block open, so the model reasons in
prose instead and runs to the token cap. The flag makes the server close the
block the same way. A client can still override it per request with
`chat_template_kwargs`, which is what `tools/smoke_test.py --no-thinking`
does.

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

- `shape=array`. Under `json_object` vLLM will not give you one: its grammar
  forces a JSON object, so the model returns a single ad dict instead of the
  array it was trained to emit, and every call scores as a compliance
  failure. Confirmed on the first checkpoint. Set `response_format = "text"`
  in the benchmark config, and note that deleting the line instead falls back
  to the `json_object` default.
- `finish=stop`, not `finish=length`. A run to the token cap means the model
  never emitted a stop token, which at roughly 30 tokens per second is minutes
  per call and reads as a hang from the harness side. Prose that starts "The
  user wants me to" is the thinking-template mismatch above, not a bad
  fine-tune; serve with `--default-chat-template-kwargs` and retest.
- Latency in seconds. A healthy window takes 4 to 5 seconds and about 100
  output tokens, which puts a 171 window single-trial run near 15 minutes. If
  one sequential window takes minutes, the harness will time out no matter how
  the concurrency is set. Pass `--max-tokens 512` while iterating so a runaway
  surfaces in a minute instead of ten.

## 6. Score with the MinusPod benchmark harness

Do not run this inside the MinusPod checkout. The harness writes to
`<package root>/results`, which is tracked in git and holds the published
multi-model history, so a local run rewrites those files and merges local
rows into the shared report. Copy the harness out first:

```sh
tools/setup_isolated_benchmark.sh <minuspod> ~/segue-benchmark
cp benchmark.local.toml.example ~/segue-benchmark/benchmark.toml
$EDITOR ~/segue-benchmark/benchmark.toml     # set the vLLM base_url
cd ~/segue-benchmark && uv sync && uv add flask
```

The Flask dependency is needed because the harness imports shared modules
from MinusPod's `src/`, which pull it in.

```sh
export SEGUE_LOCAL_API_KEY=dummy
uv run benchmark run --dry-run
uv run benchmark run --snapshot prompts/2026-08.txt
```

Check the dry-run count before committing to a run. One model at one trial
over the 14 episode corpus is 171 calls. A number in the tens of thousands
means the shipped multi-model roster is still in play, which would spend real
money on hosted providers, so fix the config rather than start the run.

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
