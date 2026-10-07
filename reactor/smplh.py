"""Minimal SMPL+H skeleton forward kinematics (body joints only, no mesh).

AMASS is already Z-up. In the SMPL canonical (rest) frame the body is Y-up, faces +Z and
its left side is +X; ``CANON_TO_WORLD`` maps that to the robot convention (Z-up, facing +X,
left = +Y) so a zero-pose source lines up with the robot nominal configuration.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from reactor import DATA
from reactor.rotations import exp_map

NUM_BODY_JOINTS = 22
JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
    "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
CANON_TO_WORLD = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@lru_cache(maxsize=3)
def _model(gender: str):
    gender = gender if gender in ("male", "female") else "neutral"
    m = np.load(DATA / "body_models" / "smplh" / gender / "model.npz")
    return (
        m["v_template"].astype(np.float64),
        m["shapedirs"].astype(np.float64),
        m["J_regressor"][:NUM_BODY_JOINTS].astype(np.float64),
    )


def rest_joints(gender: str, betas: np.ndarray) -> np.ndarray:
    """(22, 3) rest-pose joint locations in the SMPL canonical frame."""
    v_template, shapedirs, J_reg = _model(gender)
    nb = min(len(betas), shapedirs.shape[-1])
    v = v_template + shapedirs[..., :nb] @ betas[:nb]
    return J_reg @ v


def forward_kinematics(rest: torch.Tensor, root_trans: torch.Tensor, aa: torch.Tensor):
    """Global joint positions/rotations.

    rest: (22, 3) canonical rest joints; root_trans: (T, 3); aa: (T, 22, 3) axis-angle.
    Returns pos (T, 22, 3) and rot (T, 22, 3, 3) in the AMASS world frame. The global rotation
    of a joint maps the canonical frame to world, so a zero pose gives rot == root_orient.
    """
    T = aa.shape[0]
    local = exp_map(aa)
    offsets = rest.clone()
    offsets[1:] = rest[1:] - rest[PARENTS[1:]]
    rot = [local[:, 0]]
    pos = [root_trans + rest[0]]
    for j in range(1, NUM_BODY_JOINTS):
        p = PARENTS[j]
        rot.append(rot[p] @ local[:, j])
        pos.append(pos[p] + (rot[p] @ offsets[j].expand(T, 3).unsqueeze(-1)).squeeze(-1))
    return torch.stack(pos, 1), torch.stack(rot, 1)
