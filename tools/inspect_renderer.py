"""Print the exact prompt a Tinker renderer builds, and what vLLM would build.

Training and serving have to agree on the assistant prefix. This shows both
so the difference is visible instead of inferred.

Usage:
    uv run python tools/inspect_renderer.py
    uv run python tools/inspect_renderer.py --renderers qwen3_5 qwen3_5_disable_thinking
"""
import argparse

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
MESSAGES = [
    {'role': 'system', 'content': 'SYSTEM'},
    {'role': 'user', 'content': 'USER'},
]


def show_tinker(model, names):
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model)
    for name in names:
        try:
            renderer = renderers.get_renderer(name, tokenizer)
        except Exception as e:
            print(f"\n=== tinker renderer {name}: unavailable ({e}) ===")
            continue
        prompt = renderer.build_generation_prompt(MESSAGES)
        text = tokenizer.decode(list(prompt.to_ints()))
        print(f"\n=== tinker renderer {name} ===")
        print(repr(text[-240:]))
        try:
            print('stop sequences:', renderer.get_stop_sequences())
        except Exception:
            pass


def show_hf(model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    for thinking in (True, False):
        try:
            text = tokenizer.apply_chat_template(
                MESSAGES, add_generation_prompt=True, tokenize=False,
                enable_thinking=thinking)
        except TypeError:
            text = tokenizer.apply_chat_template(
                MESSAGES, add_generation_prompt=True, tokenize=False)
            print('(template takes no enable_thinking kwarg)')
        print(f"\n=== hf chat template, enable_thinking={thinking} ===")
        print(repr(text[-240:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--renderers', nargs='*',
                    default=['qwen3_5', 'qwen3_5_disable_thinking'])
    args = ap.parse_args()

    show_tinker(args.model, args.renderers)
    show_hf(args.model)
    print('\nThe tail of the disable-thinking renderer is what the model was '
          'trained to continue from. vLLM must produce the same tail, or the '
          'model answers in prose.')


if __name__ == '__main__':
    main()
