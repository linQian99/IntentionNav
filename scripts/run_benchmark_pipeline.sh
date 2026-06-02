#!/usr/bin/env bash
# Run collection, annotation, and benchmark build in sequence.
#
# Usage:
#   bash scripts/run_benchmark_pipeline.sh
#   bash scripts/run_benchmark_pipeline.sh 0 20
#   SKIP_CAPTURE=1 bash scripts/run_benchmark_pipeline.sh
#   SKIP_ANNOTATE=1 bash scripts/run_benchmark_pipeline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_CAPTURE="${SKIP_CAPTURE:-0}"
SKIP_ANNOTATE="${SKIP_ANNOTATE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"

if [ "$SKIP_CAPTURE" = "0" ]; then
    bash scripts/batch_capture_1gpu.sh "$@"
fi

if [ "$SKIP_ANNOTATE" = "0" ]; then
    bash scripts/batch_annotate.sh "$@"
fi

if [ "$SKIP_BUILD" = "0" ]; then
    bash scripts/build_benchmark.sh
fi
