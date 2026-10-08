"""Render AMASS clips: the original human next to the ReActor-retargeted G1.

The robot is the trained policy tracking the clip in MuJoCo Warp, using the trained
retargeting parameters *including* the clip's learned per-motion offset p_z (these clips are
part of the training set). The human is the original AMASS motion on its own SMPL+H body
(subject gender and betas), shifted sideways by ``--gap`` metres so the two do not overlap.
The camera follows the midpoint of the pair.

    DISPLAY=:1 MUJOCO_GL=glfw uv run python scripts/render_amass.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from reactor import DATA, ROOT
from reactor.render import (
    BLUE, HumanMesh, add_grid, drawtext, encode, look_at_camera, rollout, scene_model, set_robot_qpos,
    visual_option,
)
from reactor.source import resample

FPS = 30.0
# A spread of motion types, all retargeted successfully in the evaluation.
DEFAULT_CLIPS = [
    "ACCAD/Female1Walking_c3d/B10 - walk turn left (45)_poses.npz",
    "ACCAD/Male2Running_c3d/C1 - stand to run_poses.npz",
    "KIT/572/dance_chacha01_poses.npz",
    "KIT/200/KickHuefthoch01_poses.npz",
    "KIT/3/jump_back02_poses.npz",
    "KIT/3/kneel_down_to_crawl01_poses.npz",
    "KIT/3/squat01_poses.npz",
    "KIT/3/throw_left01_poses.npz",
]


def load_human(path: Path):
    """AMASS clip resampled to the video rate: (T, 22, 3) axis-angle, (T, 3) trans, mesh."""
    d = np.load(path)
    src_fps = float(d["mocap_framerate"])
    T = d["poses"].shape[0]
    T = min(T, int(60.0 * src_fps))  # prep_amass caps clips at 60 s
    aa = torch.tensor(d["poses"][:T, :66].reshape(T, 22, 3), dtype=torch.float64)
    trans = torch.tensor(d["trans"][:T], dtype=torch.float64)
    # resample() targets 50 Hz; get 30 Hz by pretending the source is 50/30 times faster
    trans, aa = resample(trans, aa, src_fps * 50.0 / FPS)
    mesh = HumanMesh("human", str(d["gender"]), d["betas"].astype(np.float64))
    return aa.numpy(), trans.numpy(), mesh


def human_floor(mesh: HumanMesh, aa: np.ndarray, trans: np.ndarray) -> float:
    """Lowest skinned-vertex height over a sample of frames (the clip's floor level)."""
    lows = []
    for i in np.linspace(0, len(aa) - 1, min(len(aa), 40)).astype(int):
        pos, quat = mesh.bones(aa[i], trans[i])
        R = Rotation.from_quat(quat, scalar_first=True).as_matrix()
        # bone transform: x -> R (x - rest_j) + pos_j, blended by skin weights
        local = mesh.v[:, None, :] - mesh.rest[None]
        verts = np.einsum("vj,jab,vjb->va", mesh.w, R, local) + mesh.w @ pos
        lows.append(verts[:, 2].min())
    return float(np.min(lows))


def render_clip(name: str, qpos50: np.ndarray, failed: bool, out_dir: Path, gap: float, width: int, height: int):
    aa, trans, human = load_human(DATA / "amass" / name)
    n = min(len(aa), int(len(qpos50) / 50.0 * FPS))
    aa, trans = aa[:n], trans[:n].copy()
    trans[:, 2] -= human_floor(human, aa, trans)  # stand the human on the grid
    # Place the human beside the robot: perpendicular to the robot's initial heading.
    heading = qpos50[0, 3:7]
    yaw = np.arctan2(2 * (heading[0] * heading[3] + heading[1] * heading[2]),
                     1 - 2 * (heading[2] ** 2 + heading[3] ** 2))
    side = np.array([-np.sin(yaw), np.cos(yaw), 0.0])  # robot's left
    # start the human's pelvis gap metres to the left of the robot's start position
    pelvis0 = human.bones(aa[0], trans[0])[0][0]
    trans[:, :2] += qpos50[0, :2] - pelvis0[:2] + side[:2] * gap

    model = scene_model([(human, BLUE)])
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    opt = visual_option()
    frame_dir = out_dir / f"_frames_{Path(name).stem}"
    shutil.rmtree(frame_dir, ignore_errors=True)
    frame_dir.mkdir(parents=True)

    # camera: three-quarter view from the robot's front-right, following the pair's midpoint
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    view = 2.6 * fwd - 2.4 * side  # closer than the seamless shot: only the pair is in view
    mids = []
    for i in range(n):
        set_robot_qpos(data, qpos50, i / FPS * 50.0)
        data.mocap_pos[:], data.mocap_quat[:] = human.bones(aa[i], trans[i])
        mids.append(0.5 * (data.qpos[:2] + data.mocap_pos[0, :2]))
    mids = np.array(mids)
    k = int(FPS)  # 1 s moving average keeps the camera steady
    smooth = np.stack([np.convolve(np.pad(mids[:, c], (k // 2, k - 1 - k // 2), mode="edge"),
                                   np.ones(k) / k, mode="valid") for c in range(2)], 1)
    for i in range(n):
        set_robot_qpos(data, qpos50, i / FPS * 50.0)
        data.mocap_pos[:], data.mocap_quat[:] = human.bones(aa[i], trans[i])
        mujoco.mj_kinematics(model, data)
        centre = np.r_[smooth[i], 0.85]
        look_at_camera(cam, centre, centre + view + [0, 0, 0.6])
        renderer.update_scene(data, cam, opt)
        add_grid(renderer.scene, center=smooth[i], half=6.0)
        Image.fromarray(renderer.render()).save(frame_dir / f"frame_{i:05d}.png")
    renderer.close()

    title = name.removesuffix("_poses.npz").replace("_c3d", "")
    labels = [
        drawtext(title, "24", "20", 24),
        drawtext("human (AMASS)  |  G1 (ReActor policy, simulated)", "24", "60", 20),
    ]
    if failed:
        labels.append(drawtext("robot fell - pose frozen after failure", "24", "96", 20))
    out = out_dir / f"{Path(name).parent.name}_{Path(name).stem.removesuffix('_poses')}.mp4".replace(" ", "_")
    return encode(frame_dir, out, FPS, filters=labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="reactor_g1")
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS, help="AMASS paths relative to data/amass")
    ap.add_argument("--out", type=Path, default=ROOT / "renders" / "amass")
    ap.add_argument("--gap", type=float, default=1.2, help="sideways human-robot distance [m]")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)

    wanted = set(args.clips)
    data = DATA / "processed" / "amass_g1.npz"
    available = set(str(n) for n in np.load(data)["names"])
    missing = wanted - available
    if missing:
        print(f"  skipping clips not in {data} (filtered out or not downloaded): {sorted(missing)}")
    wanted &= available
    if not wanted:
        ap.error("none of the requested clips are in the processed dataset")
    print(f"[1/2] simulating {len(wanted)} clips with '{args.run}'")
    names, qpos, failed = rollout(args.run, data, dev, names_filter=lambda n: n in wanted, trained_pz=True)
    print("[2/2] rendering")
    for name, q, f in zip(names, qpos, failed):
        path = render_clip(name, q, bool(f), args.out, args.gap, args.width, args.height)
        print(f"  {'FELL ' if f else 'ok   '} {path}")


if __name__ == "__main__":
    main()
