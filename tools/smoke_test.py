"""Send real windows to a served checkpoint and report what comes back.

Runs one window at a time, with and without the JSON response_format the
benchmark harness sets, so a stall or a malformed answer is attributable
before spending an hour on a full benchmark run.

Usage:
    uv run python tools/smoke_test.py --base-url http://host:8123/v1 --n 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, iter_examples, load_prompt  # noqa: E402


def load_windows(limit, only_with_ads):
    out = []
    for _, _, ex in iter_examples():
        if only_with_ads and not ex['completion']:
            continue
        out.append(ex)
        if len(out) >= limit:
            break
    return out


def call(base_url, model, system, user, max_tokens, timeout, response_format):
    payload = {
        'model': model,
        'temperature': 0.0,
        'max_tokens': max_tokens,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
    }
    if response_format:
        payload['response_format'] = {'type': response_format}
    t0 = time.perf_counter()
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                      json=payload, timeout=timeout,
                      headers={'Authorization': 'Bearer dummy'})
    elapsed = time.perf_counter() - t0
    if r.status_code != 200:
        return {'ok': False, 'status': r.status_code,
                'body': r.text[:400], 'elapsed': elapsed}
    body = r.json()
    choice = body['choices'][0]
    return {
        'ok': True,
        'text': choice['message']['content'] or '',
        'finish_reason': choice.get('finish_reason'),
        'usage': body.get('usage', {}),
        'elapsed': elapsed,
    }


def classify(text):
    """What shape did the model actually answer with?"""
    stripped = (text or '').strip()
    if not stripped:
        return 'empty', None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return 'not_json', None
    if isinstance(parsed, list):
        return 'array', parsed
    if isinstance(parsed, dict):
        return 'object', parsed
    return 'other_json', parsed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url', required=True, help='e.g. http://host:8123/v1')
    ap.add_argument('--model', default='/model',
                    help='model id as vLLM registered it')
    ap.add_argument('--n', type=int, default=3, help='windows to send')
    ap.add_argument('--max-tokens', type=int, default=4096,
                    help='matches the harness default')
    ap.add_argument('--timeout', type=int, default=600)
    ap.add_argument('--ads-only', action='store_true',
                    help='only send windows that contain ads')
    ap.add_argument('--skip-plain', action='store_true',
                    help='only test the json_object path')
    args = ap.parse_args()

    examples = load_windows(args.n, args.ads_only)
    if not examples:
        raise SystemExit('no examples found; run tools/extract.py first')

    modes = ['json_object'] if args.skip_plain else ['json_object', None]
    system_cache = {}
    failures = 0

    for ex in examples:
        ref = ex['prompt']['system']
        if ref not in system_cache:
            system_cache[ref] = load_prompt(ref)
        print(f"\n=== {ex['id']} (expects {len(ex['completion'])} ad(s)) ===")
        for mode in modes:
            label = mode or 'no response_format'
            res = call(args.base_url, args.model, system_cache[ref],
                       ex['prompt']['user'], args.max_tokens, args.timeout, mode)
            if not res['ok']:
                failures += 1
                print(f"  {label}: HTTP {res['status']} after "
                      f"{res['elapsed']:.1f}s -> {res['body']}")
                continue
            shape, parsed = classify(res['text'])
            usage = res['usage']
            n = len(parsed) if isinstance(parsed, list) else 'n/a'
            print(f"  {label}: {res['elapsed']:.1f}s  shape={shape}  ads={n}  "
                  f"finish={res['finish_reason']}  "
                  f"out_tokens={usage.get('completion_tokens')}  "
                  f"in_tokens={usage.get('prompt_tokens')}")
            if shape != 'array':
                failures += 1
                print(f"    raw: {res['text'][:300]!r}")
            if res['finish_reason'] == 'length':
                failures += 1
                print('    hit max_tokens: the model never emitted a stop token')

    print(f"\n{failures} problem(s) across {len(examples)} window(s) "
          f"x {len(modes)} mode(s)")
    print('The harness parses a JSON array. Any shape other than "array" '
          'lowers JSON compliance; finish=length means generation ran away.')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
