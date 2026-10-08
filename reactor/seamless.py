"""Seamless Interaction clips as ReActor source motions.

Mirrors holosoma's seamless scene conventions so the robot lands where the subject it
replaces stood:
  * subjects of an interaction are the two file ids sorted by name; subject 0 sits at the
    origin facing +x, subject 1 at ``face_to_face_dist`` along +x facing -x,
  * the camera frame becomes Z-up through Rx(-90 deg) (a rotation, not a reflection),
  * the dataset root translation is monocular depth and unusable, so it is discarded;
    the pelvis is held at the slot's xy and its height comes from the floor contact.
The npz carries no betas, so every subject uses the zero-shape neutral body.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from reactor.smplh import NUM_BODY_JOINTS, PARENTS, rest_joints

SEAMLESS_ROOT = Path(os.environ.get("SEAMLESS_ROOT", Path(__file__).resolve().parents[1] / "data" / "seamless_interaction"))
SEAMLESS_FPS = 30.0
FACE_TO_FACE_DIST = 1.4
CAM_TO_WORLD = Rotation.from_euler("x", -90, degrees=True).as_matrix()
SEGMENT_RE = re.compile(r"^(V\d+_S\d+_I\d+)_S(\d)_(\d+)_(\d+)$")


@dataclass(frozen=True)
class Segment:
    interaction_id: str
    subject: int  # subject replaced by the robot
    start: int  # 30 Hz frame indices into the clip
    end: int

    @property
    def name(self) -> str:
        return f"{self.interaction_id}_S{self.subject}_{self.start}_{self.end}"

    @classmethod
    def parse(cls, name: str) -> Segment:
        m = SEGMENT_RE.match(name)
        if not m:
            raise ValueError(f"not a segment name: {name}")
        return cls(m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))


def find_pair(interaction_id: str, root: Path = SEAMLESS_ROOT) -> tuple[list[str], list[Path]]:
    """Both participants' file ids (sorted, as holosoma orders subjects) and archive dirs."""
    hits = sorted(root.glob(f"*/*/*/*/smplh/global_orient/{interaction_id}_P*.npy"))
    if len(hits) != 2:
        raise FileNotFoundError(f"expected 2 participants for {interaction_id}, found {len(hits)}")
    ids = [h.stem for h in hits]
    order = np.argsort(ids)
    return [ids[i] for i in order], [hits[i].parents[2] for i in order]


def find_audio(file_id: str, archive: Path) -> Path | None:
    p = archive / f"{file_id}.wav"
    return p if p.exists() else None


def load_pose(archive: Path, file_id: str, start: int, end: int):
    """global_orient (T, 3) and body_pose (T, 21, 3), camera frame."""
    go = np.load(archive / "smplh" / "global_orient" / f"{file_id}.npy", mmap_mode="r")
    bp = np.load(archive / "smplh" / "body_pose" / f"{file_id}.npy", mmap_mode="r")
    end = min(end, len(go))
    return np.asarray(go[start:end], np.float64), np.asarray(bp[start:end], np.float64)


def placed_axis_angle(global_orient: np.ndarray, body_pose: np.ndarray, subject: int,
                      rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a camera-frame clip into its slot.

    Returns the (T, 22, 3) axis-angle pose with the placed root orientation, and the
    (3, 3) world rotation that was applied (Z-up + slot yaw).
    """
    T = len(global_orient)
    root = Rotation.from_rotvec(global_orient)
    world = Rotation.from_matrix(CAM_TO_WORLD) * root
    # Facing from the hip axis of the first frame (holosoma: body_forward on ref frame 0).
    hips = world[0].apply(rest[[1, 2]] - rest[0])
    left = hips[0] - hips[1]
    left[2] = 0.0
    left /= np.linalg.norm(left)
    fwd = np.cross(left, [0.0, 0.0, 1.0])
    target = 0.0 if subject == 0 else np.pi
    yaw = Rotation.from_euler("z", target - np.arctan2(fwd[1], fwd[0]))
    placed_root = yaw * world
    aa = np.zeros((T, NUM_BODY_JOINTS, 3))
    aa[:, 0] = placed_root.as_rotvec()
    aa[:, 1:] = body_pose[:, : NUM_BODY_JOINTS - 1]
    return aa, (yaw * Rotation.from_matrix(CAM_TO_WORLD)).as_matrix()


def slot_translation(aa: np.ndarray, rest: np.ndarray, subject: int, h_src: float,
                     face_to_face_dist: float = FACE_TO_FACE_DIST) -> np.ndarray:
    """Root translation keeping the pelvis at the slot xy and the lowest joint near the floor.

    The seamless clip has no usable translation. We pin the pelvis horizontally and drop it
    vertically per frame so the lowest foot joint sits at its rest height above the soles,
    with temporal smoothing so the vertical motion is not jittery.
    """
    import torch

    from reactor.smplh import forward_kinematics

    T = len(aa)
    rest_t = torch.tensor(rest, dtype=torch.float64)
    zero = torch.zeros(T, 3, dtype=torch.float64)
    pos, _ = forward_kinematics(rest_t, zero, torch.tensor(aa))
    pos = pos.numpy()
    # rest height of ankle/foot joints above the soles
    rest_up = rest[:, 1] - rest[0, 1] + h_src
    foot_ids = [7, 8, 10, 11]
    lowest = (pos[:, foot_ids, 2] - rest_up[foot_ids]).min(-1)  # sole height relative to root
    z = -lowest
    k = max(1, int(round(0.2 * SEAMLESS_FPS)))
    kernel = np.ones(k) / k
    z = np.convolve(np.pad(z, (k // 2, k - 1 - k // 2), mode="edge"), kernel, mode="valid")
    trans = np.zeros((T, 3))
    trans[:, 0] = (0.0 if subject == 0 else face_to_face_dist) - pos[0, 0, 0]
    trans[:, 1] = -pos[0, 0, 1]
    trans[:, 2] = z
    return trans


def zero_shape_rest() -> np.ndarray:
    return rest_joints("neutral", np.zeros(16))


__all__ = [
    "SEAMLESS_FPS", "FACE_TO_FACE_DIST", "Segment", "find_pair", "find_audio", "load_pose",
    "placed_axis_angle", "slot_translation", "zero_shape_rest", "PARENTS",
]
