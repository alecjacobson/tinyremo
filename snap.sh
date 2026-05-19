#!/bin/bash
# Usage: ./snap.sh <label>
# Runs the benchmark and saves output to snapshots/<label>.txt
set -e
if [ -z "$1" ]; then echo "Usage: $0 <label>"; exit 1; fi
REPO=$(cd "$(dirname "$0")" && pwd)
BENCH="$REPO/build/benchmark"
SNAP_DIR="$REPO/snapshots"
mkdir -p "$SNAP_DIR"
echo "# $(date)" > "$SNAP_DIR/${1}.txt"
echo "# $(uname -sr)" >> "$SNAP_DIR/${1}.txt"
echo "Running benchmark..."
"$BENCH" | tee -a "$SNAP_DIR/${1}.txt"
echo "Saved → snapshots/${1}.txt"
