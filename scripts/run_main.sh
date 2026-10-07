#!/usr/bin/env bash
# Full ReActor run followed by the RL-only ablation (Fig. 6), 20k iterations each.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
uv run python scripts/train.py --run reactor_g1 2>&1 | grep --line-buffered -vE "^Module |limit reached|duplicate lines" > logs/reactor_g1.log
uv run python scripts/train.py --run rlonly_g1 --no-bilevel 2>&1 | grep --line-buffered -vE "^Module |limit reached|duplicate lines" > logs/rlonly_g1.log
