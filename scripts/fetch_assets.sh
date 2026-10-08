#!/usr/bin/env bash
# Fetch the Unitree G1 (29 DoF) MJCF + meshes from the public holosoma repo into assets/g1.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=https://github.com/amazon-far/holosoma.git
COMMIT=6e146b0af5d7cd8a39b8bb2ed05b977cf70445d3  # files verified identical to the ones used for training
SRC=src/holosoma/holosoma/data/robots/g1

if [ -f assets/g1/scenes/scene_g1_29dof_wbt_plane.xml ]; then
    echo "assets/g1 already present"
    exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git -C "$tmp" init -q
git -C "$tmp" remote add origin "$REPO"
git -C "$tmp" sparse-checkout set "$SRC"
git -C "$tmp" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git -C "$tmp" checkout -q FETCH_HEAD
mkdir -p assets
cp -r "$tmp/$SRC" assets/g1
echo "fetched G1 assets -> assets/g1"
