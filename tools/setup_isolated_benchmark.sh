#!/usr/bin/env bash
# Stand up a copy of MinusPod's benchmark harness that scores only the local
# model, leaving the MinusPod checkout untouched.
#
# The harness writes results to <package root>/results, which is tracked in
# git and holds the published multi-model history. Running there would rewrite
# those files and merge local rows into the shared report, so the run gets its
# own tree with an empty results directory.
#
# The copy keeps the benchmarks/llm depth on purpose: benchmark/__init__.py
# finds MinusPod's shared src/ at parents[3].parent/"src", so a flattened
# copy resolves that to the wrong place and the shared imports fail. The
# symlink puts the real src/ exactly where that walk expects it.
#
# Usage:
#   tools/setup_isolated_benchmark.sh <minuspod-repo> [dest]
set -euo pipefail

MINUSPOD="$(cd "${1:?usage: $0 <minuspod-repo> [dest]}" && pwd)"
DEST="${2:-$HOME/segue-benchmark}"
SRC="$MINUSPOD/benchmarks/llm"

[ -d "$SRC" ] || { echo "not found: $SRC" >&2; exit 1; }
[ -d "$MINUSPOD/src" ] || { echo "not found: $MINUSPOD/src" >&2; exit 1; }
[ -e "$DEST" ] && { echo "destination exists: $DEST" >&2; exit 1; }

HARNESS="$DEST/benchmarks/llm"
mkdir -p "$HARNESS"
ln -s "$MINUSPOD/src" "$DEST/src"

# Source and corpus, but none of the recorded runs.
for item in src pyproject.toml uv.lock prompts scripts README.md; do
    [ -e "$SRC/$item" ] && cp -R "$SRC/$item" "$HARNESS/"
done
mkdir -p "$HARNESS/data" "$HARNESS/results/raw"
cp -R "$SRC/data/corpus" "$HARNESS/data/corpus"
[ -d "$SRC/data/pricing_snapshots" ] && cp -R "$SRC/data/pricing_snapshots" "$HARNESS/data/"

TOOLS="$(cd "$(dirname "$0")" && pwd)"
echo "isolated harness: $HARNESS"
echo
echo "Next:"
echo "  cp $TOOLS/../benchmark.local.toml.example $HARNESS/benchmark.toml"
echo "  \$EDITOR $HARNESS/benchmark.toml     # set the vLLM base_url"
echo "  cd $HARNESS && uv sync && uv add flask"
echo "  export SEGUE_LOCAL_API_KEY=dummy"
echo "  uv run benchmark run --dry-run       # expect 171"
