#!/usr/bin/env bash
# Fetch the robot MJCFs + meshes into assets/:
#   assets/g1  Unitree G1 (29 DoF), from the public holosoma repo
#   assets/k1  Booster K1 (22 DoF), from the public BoosterRobotics/booster_assets repo
set -euo pipefail
cd "$(dirname "$0")/.."

# fetch <name> <repo> <commit> <dir in repo> <file that marks a complete copy>
fetch() {
    local name=$1 repo=$2 commit=$3 src=$4 marker=$5
    if [ -f "assets/$name/$marker" ]; then
        echo "assets/$name already present"
        return
    fi
    local tmp
    tmp=$(mktemp -d)
    git -C "$tmp" init -q
    git -C "$tmp" remote add origin "$repo"
    git -C "$tmp" sparse-checkout set "$src"
    git -C "$tmp" fetch -q --depth 1 --filter=blob:none origin "$commit"
    git -C "$tmp" checkout -q FETCH_HEAD
    mkdir -p assets
    cp -r "$tmp/$src" "assets/$name"
    rm -rf "$tmp"
    echo "fetched $name assets -> assets/$name"
}

# files verified identical to the ones used for training
fetch g1 https://github.com/amazon-far/holosoma.git 6e146b0af5d7cd8a39b8bb2ed05b977cf70445d3 \
    src/holosoma/holosoma/data/robots/g1 scenes/scene_g1_29dof_wbt_plane.xml
fetch k1 https://github.com/BoosterRobotics/booster_assets.git 12a97516f57469078707afb587a5f84c1cb110d1 \
    robots/K1 K1_22dof.xml
