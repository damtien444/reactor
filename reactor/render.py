"""Shared rendering: policy rollouts, SMPL+H meshes as MuJoCo skins, scene, camera, encoding.

Rendering needs an OpenGL context. EGL is unavailable on this machine, so callers run with
``DISPLAY=:1 MUJOCO_GL=glfw``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from reactor import DATA, ROOT
from reactor.env import EnvCfg, ReActorEnv
from reactor.motion import MotionLib, RetargetParams
from reactor.ppo import ActorCritic, PPOCfg
from reactor.robot import SCENE_XML
from reactor.smplh import NUM_BODY_JOINTS, PARENTS

SUBTITLE_STYLE = ("FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H90000000,"
                  "BorderStyle=3,Outline=1,Shadow=0,MarginV=24")
ORANGE = [0.93, 0.55, 0.3, 1.0]
BLUE = [0.35, 0.65, 0.95, 1.0]


# ---------------------------------------------------------------------- simulation
@torch.no_grad()
def rollout(run: str, data: Path, device, names_filter=None, trained_pz: bool = False, seed: int = 0):
    """Roll out the trained policy once on every selected motion, from frame 0.

    Returns the 50 Hz qpos per motion and whether the robot failed. ``trained_pz`` uses the
    learned per-motion vertical offsets, which only exist for motions of the training set
    (``data`` must then be the training data); otherwise p_z is 0.
    """
    ck = torch.load(ROOT / "runs" / run / "model.pt", map_location=device, weights_only=False)
    lib = MotionLib(data, device, names_filter)
    uc = ck["upper_cfg"]
    params = RetargetParams(lib.x_nom.shape[1], lib.num_motions, uc["d_pos"], uc["d_ori"], uc["d_z"]).to(device)
    trained = ck["params"]
    params.p_pos.copy_(trained["p_pos"])
    params.p_ori.copy_(trained["p_ori"])
    if trained_pz:
        params.p_z.copy_(trained["p_z"][torch.as_tensor(lib.indices, device=device)])
    ecfg = EnvCfg(**(ck["env_cfg"] | {"num_envs": lib.num_motions, "max_episode_steps": 10**9}))
    env = ReActorEnv(ecfg, lib, params, device)
    ac = ActorCritic(ck["num_obs"], ck["num_actions"], PPOCfg()).to(device)
    ac.load_state_dict(ck["ac"])
    ac.eval()

    M = lib.num_motions
    motions = torch.arange(M, device=device)
    torch.manual_seed(seed)  # the initial joint configuration is randomized (Sec. 6)
    env.reset(motions, motions, torch.zeros_like(motions))
    obs = env._observe()
    L = lib.lengths
    T = int(L.max()) + ecfg.init_steps
    qpos = torch.zeros(M, int(L.max()), env.qpos.shape[1], device=device)
    failed = torch.zeros(M, dtype=torch.bool, device=device)
    for k in range(T):
        obs, _, term, _, _ = env.step(ac.act_inference(obs))
        t = k + 1 - ecfg.init_steps
        if t < 0:
            continue
        active = (t < L) & ~failed
        # an env that terminated was auto-reset; keep its last valid pose from here on
        failed |= term & active
        tt = min(t, qpos.shape[1] - 1)
        qpos[:, tt] = torch.where((active & ~term).unsqueeze(-1), env.qpos, qpos[:, max(tt - 1, 0)])
    return lib.names, [qpos[i, : int(L[i])].cpu().numpy() for i in range(M)], failed.cpu().numpy()


# ---------------------------------------------------------------------- scene
class HumanMesh:
    """SMPL+H body as a MuJoCo skin whose bones are mocap bodies.

    MuJoCo skins the mesh itself (linear blend skinning, no pose blendshapes), so the human
    shades like the robot. Hand-joint weights are folded into the wrists.
    """

    def __init__(self, prefix: str, gender: str = "neutral", betas: np.ndarray | None = None):
        gender = gender if gender in ("male", "female") else "neutral"
        m = np.load(DATA / "body_models" / "smplh" / gender / "model.npz")
        self.prefix = prefix
        v = m["v_template"]
        if betas is not None:
            nb = min(len(betas), m["shapedirs"].shape[-1])
            v = v + m["shapedirs"][..., :nb] @ betas[:nb]
        self.v = v
        self.faces = m["f"].astype(np.int64)
        w = m["weights"]
        self.w = w[:, :NUM_BODY_JOINTS].copy()
        self.w[:, 20] += w[:, 22:37].sum(1)
        self.w[:, 21] += w[:, 37:52].sum(1)
        self.rest = m["J_regressor"][:NUM_BODY_JOINTS] @ self.v

    def add_to(self, spec: mujoco.MjSpec, rgba) -> None:
        for j in range(NUM_BODY_JOINTS):
            spec.worldbody.add_body(name=f"{self.prefix}_{j}", mocap=True)
        skin = spec.add_skin(name=self.prefix)
        skin.vert = self.v.flatten().tolist()
        skin.face = self.faces.flatten().tolist()
        skin.rgba = rgba
        skin.inflate = 0.0
        names, bindpos, bindquat, vertid, vertweight = [], [], [], [], []
        for j in range(NUM_BODY_JOINTS):
            ids = np.nonzero(self.w[:, j] > 1e-4)[0]
            names.append(f"{self.prefix}_{j}")
            # bind pose: bone frame at the rest joint, identity orientation
            bindpos += self.rest[j].tolist()
            bindquat += [1.0, 0.0, 0.0, 0.0]
            vertid.append(ids.astype(int).tolist())
            vertweight.append(self.w[ids, j].astype(float).tolist())
        skin.bodyname = names
        skin.bindpos = bindpos
        skin.bindquat = bindquat
        skin.vertid = vertid
        skin.vertweight = vertweight

    def bones(self, aa: np.ndarray, trans: np.ndarray):
        """Global bone poses (positions, wxyz quats) for one frame (same FK as reactor.smplh)."""
        R = Rotation.from_rotvec(aa)
        glob = [R[0]]
        pos = [trans + self.rest[0]]
        for j in range(1, NUM_BODY_JOINTS):
            p = PARENTS[j]
            glob.append(glob[p] * R[j])
            pos.append(pos[p] + glob[p].apply(self.rest[j] - self.rest[p]))
        quat = np.stack([g.as_quat(scalar_first=True) for g in glob])
        return np.stack(pos), quat


def scene_model(humans: list[tuple[HumanMesh, list]]) -> mujoco.MjModel:
    """G1 plus SMPL+H skins; mocap bodies are ordered as ``humans``."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    for pair in list(spec.pairs):
        spec.delete(pair)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.stat.extent = 6.0  # shadow map must cover clips that travel several metres
    spec.geom("floor").rgba = [0, 0, 0, 0]  # the grid is drawn as lines, holosoma style
    for mat in spec.materials:
        if mat.name == "silver":
            mat.rgba = [0.85, 0.85, 0.88, 1.0]
    for light in list(spec.worldbody.lights):
        spec.delete(light)
    spec.worldbody.add_light(pos=[0.7, -1.5, 4.0], dir=[0, 0.3, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.5, 0.5, 0.5])
    for human, rgba in humans:
        human.add_to(spec, rgba)
    return spec.compile()


def set_robot_qpos(data: mujoco.MjData, qpos50: np.ndarray, t50: float) -> None:
    """Interpolate a 50 Hz robot trajectory at fractional frame ``t50``."""
    j = min(int(t50), len(qpos50) - 1)
    j1 = min(j + 1, len(qpos50) - 1)
    w = t50 - int(t50)
    q = (1 - w) * qpos50[j] + w * qpos50[j1]
    q[3:7] = qpos50[j, 3:7] if w < 0.5 else qpos50[j1, 3:7]
    data.qpos[: len(q)] = q


def add_grid(scn, center=(0.7, 0.0), half=3.0, step=0.5, rgba=(0.85, 0.85, 0.85, 1.0)):
    cx, cy = np.round(np.asarray(center) / step) * step
    for x in np.arange(-half, half + 1e-9, step):
        for a, b in (((cx + x, cy - half), (cx + x, cy + half)), ((cx - half, cy + x), (cx + half, cy + x))):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                                np.array(rgba, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 1.5, np.array([*a, 0.0]), np.array([*b, 0.0]))
            scn.ngeom += 1


def look_at_camera(cam: mujoco.MjvCamera, lookat, eye) -> None:
    lookat, eye = np.asarray(lookat, float), np.asarray(eye, float)
    off = eye - lookat
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = float(np.linalg.norm(off))
    # MuJoCo's azimuth/elevation give the viewing direction (look-at -> eye is its negative)
    cam.azimuth = float(np.degrees(np.arctan2(-off[1], -off[0])))
    cam.elevation = float(-np.degrees(np.arcsin(off[2] / np.linalg.norm(off))))


def visual_option() -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 0, 0, 0, 0]  # visual meshes only
    return opt


def encode(frame_dir: Path, out: Path, fps: float, audio: Path | None = None, subtitles: Path | None = None,
           filters: list[str] | None = None) -> Path:
    """PNG sequence -> H.264, with optional audio, burned-in subtitles and extra video filters."""
    vf = list(filters or [])
    if subtitles:
        esc = str(subtitles.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        vf.append(f"subtitles='{esc}':force_style='{SUBTITLE_STYLE}'")
    args = ["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:g}", "-i", str(frame_dir / "frame_%05d.png")]
    if audio:
        args += ["-i", str(audio)]
    if vf:
        args += ["-vf", ",".join(vf)]
    if audio:
        args += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    args += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(args, check=True)
    shutil.rmtree(frame_dir, ignore_errors=True)
    return out


def drawtext(text: str, x: str, y: str, size: int = 26) -> str:
    esc = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")
    return (f"drawtext=text='{esc}':x={x}:y={y}:fontsize={size}:fontcolor=white:"
            "box=1:boxcolor=black@0.45:boxborderw=8")
