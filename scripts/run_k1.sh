#!/usr/bin/env bash
# Booster K1: preprocess AMASS, then a full ReActor run (20k iterations). Meant to run in tmux.
# Never overwrites: existing processed data is reused, an existing run directory aborts.
#   scripts/run_k1.sh [run name, default reactor_k1] [extra train.py args...]
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD

RUN=${1:-reactor_k1}
shift || true
DATA=data/processed/amass_k1.npz

if [ -e "runs/$RUN" ]; then
    echo "runs/$RUN already exists; pick another run name or use train.py --resume" >&2
    exit 1
fi

if [ -e "$DATA" ]; then
    echo "$DATA exists, reusing it"
else
    uv run python scripts/prep_amass.py --robot k1 --out "$DATA"
fi

mkdir -p logs
uv run python scripts/train.py --robot k1 --data "$DATA" --run "$RUN" "$@" 2>&1 \
    | grep --line-buffered -vE "^Module |limit reached|warn_overflow|duplicate lines" | tee "logs/$RUN.log"
