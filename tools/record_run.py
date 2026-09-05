"""Collect a finished run's artifacts into runs/ for committing.

The trainer writes runs/<run-id>/run.json, the generation eval writes
.local/eval-<run-id>.json, and export writes an export manifest. Only the
first is in git, and .local is ignored, so the numbers are lost unless they
are copied out. This does that, and writes a results document beside them.

Usage:
    uv run python tools/record_run.py --run-id r2
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT  # noqa: E402


def scrub_paths(obj, home, repo_root):
    """Replace absolute paths with placeholders, without mutating the input.

    Run manifests record the machine they ran on; a public repo does not need
    someone's username and directory layout.
    """
    if isinstance(obj, dict):
        return {k: scrub_paths(v, home, repo_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_paths(v, home, repo_root) for v in obj]
    if isinstance(obj, str):
        if obj.startswith(str(repo_root)):
            return obj.replace(str(repo_root), '<repo>', 1)
        if obj.startswith(str(home)):
            return obj.replace(str(home), '~', 1)
    return obj


def _pct(value):
    return f'{value:.4f}' if isinstance(value, float) else str(value)


def render_results(run_id, manifest, metrics, export_manifest):
    """Build the facts half of a results document. Analysis stays human."""
    args = manifest.get('args', {})
    env = manifest.get('env', {})
    truth_spans = metrics.get('tp', 0) + metrics.get('fn', 0)
    lines = [
        f'# Run {run_id}',
        '',
        f"{args.get('model', 'unknown')} at revision "
        f"`{args.get('revision', 'unknown')}`, LoRA rank {args.get('rank')}, "
        f"{args.get('epochs')} epochs, batch {args.get('batch_size')} with "
        f"{args.get('grad_accum')} accumulation steps. "
        f"{manifest.get('trainable_params', 0):,} trainable parameters.",
        '',
        '## Held-out generation eval',
        '',
        'Scored on the feed-held-out validation split, not on the benchmark',
        'corpus, so these numbers are not comparable to rows in MinusPod\'s',
        'published table. Spans on both sides are merged into contiguous',
        'breaks before matching at IoU >= 0.5.',
        '',
        '| Metric | Value |',
        '|---|---|',
        f"| JSON compliance | {_pct(metrics.get('json_compliance'))} |",
        f"| F0.5 | {_pct(metrics.get('f05'))} |",
        f"| Precision / Recall | {_pct(metrics.get('precision'))} / "
        f"{_pct(metrics.get('recall'))} |",
        f"| TP / FP / FN | {metrics.get('tp')} / {metrics.get('fp')} / "
        f"{metrics.get('fn')} |",
        f"| No-ad false positives | {metrics.get('noad_fp')} across "
        f"{metrics.get('noad_windows')} no-ad windows |",
        f"| Boundary MAE | {metrics.get('start_mae')}s start, "
        f"{metrics.get('end_mae')}s end |",
        f"| Windows scored | {metrics.get('n')} |",
        '',
        f'Sample size: {truth_spans} truth spans. One span moves F0.5 by '
        'roughly two points at this scale, so read the headline as a '
        'direction, not a measurement.',
        '',
        '## Training',
        '',
        '| Metric | Value |',
        '|---|---|',
        f"| Mean train loss | {manifest.get('train_loss')} |",
        f"| Final held-out loss | "
        f"{manifest.get('final_eval', {}).get('eval_loss')} |",
        f"| Best checkpoint | {manifest.get('best_checkpoint') or 'final'} |",
        '',
        '## Provenance',
        '',
        '| Field | Value |',
        '|---|---|',
        f"| Base revision | `{args.get('revision')}` |",
        f"| Train / val hashes | `{manifest.get('sha256', {}).get('train')}` / "
        f"`{manifest.get('sha256', {}).get('val')}` |",
        f"| segue commit | `{manifest.get('git', {}).get('segue')}` |",
        f"| torch / transformers | {env.get('torch')} / "
        f"{env.get('transformers')} |",
        f"| Device | {env.get('device', 'unknown')} |",
        f"| Seed | {args.get('seed')} |",
        '',
        '## Export',
        '',
    ]
    if export_manifest:
        matches = export_manifest.get('fixture_matches', [])
        lines += [
            f'Equivalence gate: {sum(matches)}/{len(matches)} fixtures '
            'generated identically between base plus adapter and the merged '
            f"model. Logit max diff {export_manifest.get('logit_max_diff')}.",
            '',
            '| Artifact | sha256 |',
            '|---|---|',
        ]
        for label, key in (('adapter', 'adapter_sha256'),
                           ('merged', 'merged_sha256')):
            for name, digest in sorted(export_manifest.get(key, {}).items()):
                lines.append(f'| {label} `{name}` | `{digest}` |')
    else:
        lines.append('This checkpoint was not exported, so no equivalence '
                     'gate was run and no artifact checksums exist.')
    lines += ['', '## Analysis', '',
              'Written by hand. What the numbers mean, what failed, and what '
              'to change next.', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--local-dir', default=str(REPO_ROOT / '.local'))
    ap.add_argument('--no-scrub', action='store_true',
                    help='keep absolute paths in the manifest')
    args = ap.parse_args()

    local = Path(args.local_dir)
    run_dir = REPO_ROOT / 'runs' / args.run_id
    manifest_path = run_dir / 'run.json'
    if not manifest_path.exists():
        raise SystemExit(f'{manifest_path} missing; was the run completed?')
    manifest = json.loads(manifest_path.read_text())

    eval_path = local / f'eval-{args.run_id}.json'
    if not eval_path.exists():
        raise SystemExit(f'{eval_path} missing; run tools/eval_generation.py')
    metrics = json.loads(eval_path.read_text())['metrics']

    export_path = local / 'export' / args.run_id / 'export_manifest.json'
    export_manifest = (json.loads(export_path.read_text())
                       if export_path.exists() else None)

    if not args.no_scrub:
        home = Path(os.path.expanduser('~'))
        scrubbed = scrub_paths(manifest, home, REPO_ROOT)
        if scrubbed != manifest:
            manifest_path.write_text(json.dumps(scrubbed, indent=2) + '\n')
            print(f'scrubbed absolute paths in {manifest_path.name}')
        manifest = scrubbed
        if export_manifest:
            export_manifest = scrub_paths(export_manifest, home, REPO_ROOT)

    eval_out = run_dir / 'eval.json'
    eval_out.write_text(json.dumps(metrics, indent=2) + '\n')

    results = run_dir / 'results.md'
    if results.exists():
        raise SystemExit(f'{results} exists; move it aside to regenerate')
    results.write_text(render_results(args.run_id, manifest, metrics,
                                      export_manifest))

    print(f'wrote {eval_out.relative_to(REPO_ROOT)} and '
          f'{results.relative_to(REPO_ROOT)}')
    if export_manifest is None:
        print('note: no export manifest found, the results say so')
    print(f'commit: runs/{args.run_id}/')
    print('then write the Analysis section by hand')


if __name__ == '__main__':
    main()
