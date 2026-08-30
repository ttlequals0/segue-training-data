"""LoRA-train the Segue model on Tinker from a built dataset.

Requires TINKER_API_KEY in the environment and the train extra:
    uv sync --extra train
    uv run python tools/train_tinker.py --train .local/train.jsonl

Held-out-feed val.jsonl is scored post-hoc by the MinusPod benchmark harness;
the in-run eval split here is only a loss monitor.
"""
import argparse
import asyncio
import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def pick_renderer(model_name, recommended):
    # Ad detection needs direct JSON output, so prefer the no-think template.
    if "qwen3.5" in model_name.lower() or "qwen3_5" in recommended:
        return "qwen3_5_disable_thinking"
    return recommended


def list_models():
    import tinker
    caps = tinker.ServiceClient().get_server_capabilities()
    for m in caps.supported_models:
        print(f"  - {m.model_name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', default=str(REPO_ROOT / '.local' / 'train.jsonl'))
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--renderer', default=None,
                    help='override renderer name (default: no-think variant)')
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-length', type=int, default=16384)
    ap.add_argument('--eval-size', type=int, default=16,
                    help='examples held out of --train for in-run loss monitoring')
    ap.add_argument('--log-path', default=None)
    ap.add_argument('--list-models', action='store_true')
    args = ap.parse_args()

    if 'TINKER_API_KEY' not in os.environ:
        raise SystemExit('TINKER_API_KEY not set')
    if args.list_models:
        list_models()
        return

    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    renderer = args.renderer or pick_renderer(
        args.model, model_info.get_recommended_renderer_name(args.model))
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = args.log_path or str(REPO_ROOT / '.local' / 'tinker-logs' / stamp)

    n_examples = sum(1 for _ in open(args.train, encoding='utf-8'))
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=renderer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=args.train,
        test_size=args.eval_size,
        shuffle_seed=13,
    )
    config = train.Config(
        log_path=log_path,
        model_name=args.model,
        dataset_builder=dataset_builder,
        learning_rate=args.lr,
        lora_rank=args.rank,
        num_epochs=args.epochs,
        save_every=20,
        eval_every=10,
    )

    run_config = {
        'model': args.model, 'renderer': renderer, 'rank': args.rank,
        'lr': args.lr, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'max_length': args.max_length, 'train_file': args.train,
        'train_examples': n_examples, 'started_at': stamp,
    }
    runs_dir = REPO_ROOT / 'runs'
    runs_dir.mkdir(exist_ok=True)
    (runs_dir / f'{stamp}.json').write_text(
        json.dumps(run_config, indent=2) + '\n')
    print(f'run config: runs/{stamp}.json')
    print(f'training {args.model} (renderer {renderer}, rank {args.rank}) '
          f'on {n_examples} examples, logs in {log_path}')

    try:
        asyncio.run(train.main(config))
    except Exception as e:
        if 'model' in str(e).lower():
            print(f'\nRun failed: {e}\nModels this account can train:')
            list_models()
            raise SystemExit(1)
        raise

    print('\nDone. Next steps (see tinker_cookbook.weights):')
    print('  1. weights.download(tinker_path=<sampler path from logs>, '
          'output_dir="./adapter")')
    print(f'  2. weights.build_hf_model(base_model="{args.model}", '
          'adapter_path="./adapter", output_path="./model", dtype="bfloat16")')
    print('  3. serve with vLLM and run the MinusPod benchmark against it')


if __name__ == '__main__':
    main()
