#!/usr/bin/env bash
# Stand up a copy of MinusPod's benchmark harness that scores only the local
# model, leaving the MinusPod checkout untouched.
#
# The harness writes results to <package root>/results, which is tracked in
# git and holds the published multi-model history. Running there would rewrite
# those files and merge local rows into the shared report, so the run gets its
# own tree with an empty results directory.
#
# Usage:
#   tools/setup_isolated_benchmark.sh <minuspod-repo> [dest]
set -euo pipefail

MINUSPOD="${1:?usage: $0 <minuspod-repo> [dest]}"
DEST="${2:-$HOME/segue-benchmark}"
SRC="$MINUSPOD/benchmarks/llm"

[ -d "$SRC" ] || { echo "not found: $SRC" >&2; exit 1; }
[ -e "$DEST" ] && { echo "destination exists: $DEST" >&2; exit 1; }

mkdir -p "$DEST"
# Source and corpus, but none of the recorded runs.
for item in src pyproject.toml uv.lock prompts scripts README.md; do
    [ -e "$SRC/$item" ] && cp -R "$SRC/$item" "$DEST/"
done
mkdir -p "$DEST/data" "$DEST/results/raw"
cp -R "$SRC/data/corpus" "$DEST/data/corpus"
[ -d "$SRC/data/pricing_snapshots" ] && cp -R "$SRC/data/pricing_snapshots" "$DEST/data/"

echo "isolated harness: $DEST"
echo
echo "Next:"
echo "  cp $(dirname "$0")/../benchmark.local.toml.example $DEST/benchmark.toml"
echo "  \$EDITOR $DEST/benchmark.toml     # set the vLLM base_url"
echo "  cd $DEST && uv sync && uv add flask"
echo "  export SEGUE_LOCAL_API_KEY=dummy"
echo "  uv run benchmark run --dry-run   # expect one model's worth of calls"
