"""Render Seamless Interaction segments with one subject replaced by a ReActor-driven robot.

For each segment (named ``<interaction>_S<subject>_<start>_<end>``, the holosoma report
layout) the replaced subject's SMPL+H motion becomes a ReActor source track: it is placed in
its slot like holosoma does, mapped through the trained retargeting parameters, and tracked
by the trained policy in MuJoCo Warp. The simulated robot is what gets rendered -- there is
no kinematic retargeting step. The partner stays human (SMPL mesh), and the segment's audio
and subtitles are reused from the holosoma report folder when present.

The seamless motions are new to the policy. Their per-motion vertical offset p_z was never
optimized, so it is left at 0 (z_nom from the nominal mapping); the shared per-body offsets
p_pos and p_ori are the trained ones.

Rendering uses MuJoCo's GLFW backend on the local X display (EGL is unavailable here):
    DISPLAY=:1 MUJOCO_GL=glfw uv run python scripts/render_seamless.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from reactor import ROOT
from reactor.render import (
    ORANGE, HumanMesh, add_grid, encode, look_at_camera, rollout, scene_model, set_robot_qpos, visual_option,
)
from reactor.seamless import (
    FACE_TO_FACE_DIST, SEAMLESS_FPS, Segment, find_audio, find_pair, load_pose, placed_axis_angle,
    slot_translation,
)
from reactor.source import TargetContext, body_shape, make_track, save_tracks

# Optional: a holosoma seamless report folder; its segment names, audio and subtitles are reused.
HOLOSOMA_REVIEW = Path(os.environ.get(
    "HOLOSOMA_REVIEW", ROOT.parent / "holosoma" / "src" / "holosoma_retargeting" / "reports" / "review"))


# ---------------------------------------------------------------------- simulation
def build_tracks(segments: list[Segment], out: Path, device) -> list[dict]:
    """Source tracks (robot subject) and partner poses for every segment."""
    ctx = TargetContext.create(device)
    h_src, rest = body_shape("neutral", np.zeros(16))
    tracks, names, scenes = [], [], []
    for seg in segments:
        ids, archives = find_pair(seg.interaction_id)
        go, bp = load_pose(archives[seg.subject], ids[seg.subject], seg.start, seg.end)
        aa, _ = placed_axis_angle(go, bp, seg.subject, rest)
        trans = slot_translation(aa, rest, seg.subject, h_src)
        tracks.append(make_track(ctx, rest, h_src, trans, aa, SEAMLESS_FPS, device))
        names.append(seg.name)
        partner = 1 - seg.subject
        pgo, pbp = load_pose(archives[partner], ids[partner], seg.start, seg.end)
        paa, _ = placed_axis_angle(pgo, pbp, partner, rest)
        ptrans = slot_translation(paa, rest, partner, h_src)
        scenes.append({"segment": seg, "ids": ids, "archives": archives, "partner_aa": paa,
                       "partner_trans": ptrans, "n30": len(go)})
    save_tracks(out, ctx, tracks, names)
    return scenes


def render_video(scene: dict, qpos50: np.ndarray, model: mujoco.MjModel, partner: HumanMesh, out_dir: Path,
                 width: int, height: int, audio: Path | None, subtitles: Path | None, tag: str) -> Path:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    # holosoma's three-quarter framing that holds both slots
    look_at_camera(cam, [FACE_TO_FACE_DIST / 2, 0.0, 0.9], [FACE_TO_FACE_DIST / 2 + 2.6, -3.0, 1.9])
    opt = visual_option()
    frame_dir = out_dir / f"_frames_{tag}"
    shutil.rmtree(frame_dir, ignore_errors=True)
    frame_dir.mkdir(parents=True)
    for i in range(scene["n30"]):
        set_robot_qpos(data, qpos50, i / SEAMLESS_FPS * 50.0)  # 50 Hz simulation on the 30 Hz timeline
        data.mocap_pos[:], data.mocap_quat[:] = partner.bones(scene["partner_aa"][i], scene["partner_trans"][i])
        mujoco.mj_kinematics(model, data)
        renderer.update_scene(data, cam, opt)
        add_grid(renderer.scene)
        Image.fromarray(renderer.render()).save(frame_dir / f"frame_{i:05d}.png")
    renderer.close()
    return encode(frame_dir, out_dir / f"robot_{tag}.mp4", SEAMLESS_FPS, audio, subtitles)


def segment_audio(scene: dict, out_dir: Path) -> Path | None:
    """Mix both participants' audio for the window (holosoma's audio.wav if available)."""
    seg = scene["segment"]
    holo = HOLOSOMA_REVIEW / seg.name / "audio.wav"
    if holo.exists():
        return holo
    wavs = [find_audio(fid, a) for fid, a in zip(scene["ids"], scene["archives"])]
    wavs = [w for w in wavs if w is not None]
    if not wavs:
        return None
    out = out_dir / "audio.wav"
    t0, dur = seg.start / SEAMLESS_FPS, (seg.end - seg.start) / SEAMLESS_FPS
    args = ["ffmpeg", "-v", "error", "-y"]
    for w in wavs:
        args += ["-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", str(w)]
    if len(wavs) > 1:
        args += ["-filter_complex", f"amix=inputs={len(wavs)}:normalize=0"]
    subprocess.run(args + [str(out)], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="reactor_g1")
    ap.add_argument("--segments", nargs="*", default=None,
                    help="segment names; default: every segment of the holosoma review report")
    ap.add_argument("--out", type=Path, default=ROOT / "renders" / "seamless")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    dev = torch.device("cuda")

    if args.segments:
        names = args.segments
    elif HOLOSOMA_REVIEW.is_dir():
        names = sorted(p.name for p in HOLOSOMA_REVIEW.iterdir() if p.is_dir())
    else:
        ap.error(f"no --segments given and no holosoma report at {HOLOSOMA_REVIEW} (set HOLOSOMA_REVIEW)")
    segments = [Segment.parse(n) for n in names]
    print(f"{len(segments)} segments -> {args.out}")
    data_path = args.out / "seamless_tracks.npz"
    scenes = build_tracks(segments, data_path, dev)
    print(f"[1/2] simulating ReActor policy '{args.run}'")
    names_sim, qpos, failed = rollout(args.run, data_path, dev)
    for n, f in zip(names_sim, failed):
        print(f"  {n}: {'FELL (pose frozen after failure)' if f else 'tracked'}")

    print("[2/2] rendering")
    partner = HumanMesh("partner")
    model = scene_model([(partner, ORANGE)])
    for scene, q in zip(scenes, qpos):
        seg = scene["segment"]
        out_dir = args.out / seg.name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "robot_reactor_qpos.npz", qpos=q, fps=50.0, segment=seg.name)
        subs = HOLOSOMA_REVIEW / seg.name / "subtitles.srt"
        path = render_video(scene, q, model, partner, out_dir, args.width, args.height,
                            segment_audio(scene, out_dir), subs if subs.exists() else None, "reactor")
        print(f"  {path}")


if __name__ == "__main__":
    main()
