"""Unitree G1 (29 DoF) description: MuJoCo scene, PD gains, nominal pose, correspondences.

The MJCF comes from holosoma. We rebuild the scene through MjSpec so that
  * foot/floor contacts come from regular collision filtering only (the shipped
    explicit <pair>s would duplicate every foot contact),
  * self-collision is active between all collision geoms (the paper relies on the
    simulator to prevent self-penetration), except for body pairs that already
    overlap in the rest pose,
  * motors become implicit PD position actuators so a whole control step
    (decimation sub-steps) can be captured in a single CUDA graph.
"""

from __future__ import annotations

import mujoco
import numpy as np

from reactor import ASSETS

SCENE_XML = ASSETS / "g1" / "scenes" / "scene_g1_29dof_wbt_plane.xml"

# PD gains per joint group (holosoma / BeyondMimic G1 values: 10 Hz natural freq, 2x damping)
_KP = {
    "hip_yaw": 40.179238471, "hip_roll": 99.098427777, "hip_pitch": 40.179238471,
    "knee": 99.098427777, "ankle_pitch": 28.501246196, "ankle_roll": 28.501246196,
    "waist_yaw": 40.179238471, "waist_roll": 28.501246196, "waist_pitch": 28.501246196,
    "shoulder_pitch": 14.250623098, "shoulder_roll": 14.250623098, "shoulder_yaw": 14.250623098,
    "elbow": 14.250623098, "wrist_roll": 14.250623098, "wrist_pitch": 16.778327481,
    "wrist_yaw": 16.778327481,
}
_KD = {
    "hip_yaw": 2.557889765, "hip_roll": 6.308801854, "hip_pitch": 2.557889765,
    "knee": 6.308801854, "ankle_pitch": 1.814445687, "ankle_roll": 1.814445687,
    "waist_yaw": 2.557889765, "waist_roll": 1.814445687, "waist_pitch": 1.814445687,
    "shoulder_pitch": 0.907222843, "shoulder_roll": 0.907222843, "shoulder_yaw": 0.907222843,
    "elbow": 0.907222843, "wrist_roll": 0.907222843, "wrist_pitch": 1.068141502,
    "wrist_yaw": 1.068141502,
}

# Standing pose used as mean of the randomized initial joint configuration (Sec. 6).
DEFAULT_JOINTS = {
    "left_hip_pitch_joint": -0.312, "left_knee_joint": 0.669, "left_ankle_pitch_joint": -0.363,
    "right_hip_pitch_joint": -0.312, "right_knee_joint": 0.669, "right_ankle_pitch_joint": -0.363,
    "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 0.6,
    "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2, "right_elbow_joint": 0.6,
}

# Nominal configuration (T-pose, Fig. 4) that is aligned with the SMPL rest pose.
NOMINAL_JOINTS = {
    "left_shoulder_roll_joint": np.pi / 2, "right_shoulder_roll_joint": -np.pi / 2,
    "left_elbow_joint": np.pi / 2, "right_elbow_joint": np.pi / 2,
}

# Sparse semantic correspondences (robot body, SMPL+H joint index). The first pair is the root.
CORRESPONDENCES = [
    ("pelvis", 0),
    ("left_hip_roll_link", 1),
    ("left_knee_link", 4),
    ("left_ankle_roll_link", 7),
    ("right_hip_roll_link", 2),
    ("right_knee_link", 5),
    ("right_ankle_roll_link", 8),
    ("torso_link", 9),
    ("left_shoulder_roll_link", 16),
    ("left_elbow_link", 18),
    ("left_wrist_yaw_link", 20),
    ("right_shoulder_roll_link", 17),
    ("right_elbow_link", 19),
    ("right_wrist_yaw_link", 21),
]
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def _group(joint_name: str) -> str:
    name = joint_name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
    return name


def build_model(timestep: float = 0.005, self_collision: bool = True) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    for pair in list(spec.pairs):
        spec.delete(pair)
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 10
    spec.option.ls_iterations = 10
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL

    floor = spec.geom("floor")
    floor.friction = [0.8, 0.005, 0.0001]
    floor.contype, floor.conaffinity = 1, 1
    if not self_collision:
        # robot geoms only collide with the floor
        for g in spec.geoms:
            if g.name != "floor" and g.contype:
                g.contype, g.conaffinity = 2, 1
        floor.contype, floor.conaffinity = 1, 2

    for act in spec.actuators:
        g = _group(act.target)
        kp, kd = _KP[g], _KD[g]
        lim = float(act.ctrlrange[1])
        act.set_to_position(kp=kp, kv=kd)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-lim, lim]

    model = spec.compile()
    if self_collision:
        _exclude_rest_overlaps(spec, model)
        model = spec.compile()
    return model


def _exclude_rest_overlaps(spec: mujoco.MjSpec, model: mujoco.MjModel) -> None:
    """Exclude body pairs whose collision geoms touch in default or nominal poses."""
    data = mujoco.MjData(model)
    pairs = set()
    for joints in (DEFAULT_JOINTS, NOMINAL_JOINTS, {}):
        data.qpos[:] = model.qpos0
        data.qpos[2] = 3.0  # lift away from the floor
        for name, val in joints.items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = val
        mujoco.mj_forward(model, data)
        for c in data.contact[: data.ncon]:
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if b1 and b2:
                pairs.add(tuple(sorted((b1, b2))))
    for b1, b2 in sorted(pairs):
        spec.add_exclude(bodyname1=model.body(b1).name, bodyname2=model.body(b2).name)


class RobotInfo:
    """Index bookkeeping shared by the environment, retargeting and evaluation code."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.joint_names = [model.joint(j).name for j in range(1, model.njnt)]
        self.num_joints = len(self.joint_names)
        assert model.nu == self.num_joints
        act_joint = [model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu)]
        assert act_joint == self.joint_names
        self.kp = np.array([_KP[_group(n)] for n in self.joint_names])
        self.effort = model.actuator_forcerange[:, 1].copy()
        self.joint_range = model.jnt_range[1:].copy()
        self.default_q = np.array([DEFAULT_JOINTS.get(n, 0.0) for n in self.joint_names])
        self.nominal_q = np.array([NOMINAL_JOINTS.get(n, 0.0) for n in self.joint_names])
        # BeyondMimic-style action scale: 0.25 * effort / stiffness
        self.action_scale = 0.25 * self.effort / self.kp
        self.body_ids = np.array([model.body(b).id for b, _ in CORRESPONDENCES])
        self.smpl_ids = np.array([j for _, j in CORRESPONDENCES])
        self.root_body = model.body("pelvis").id
        self.foot_ids = np.array([model.body(f).id for f in FEET])
        self.num_pairs = len(CORRESPONDENCES)

    def nominal_frames(self):
        """Global body frames (pos, rotmat) in the nominal T-pose with soles on the ground."""
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = m.qpos0
        d.qpos[7:] = self.nominal_q
        mujoco.mj_forward(m, d)
        lowest = min(
            d.geom_xpos[g, 2] - m.geom_size[g, 0]
            for g in range(m.ngeom)
            if m.geom_bodyid[g] in self.foot_ids and m.geom_contype[g]
        )
        d.qpos[2] -= lowest
        mujoco.mj_forward(m, d)
        pos = d.xpos[self.body_ids].copy()
        rot = d.xmat[self.body_ids].reshape(-1, 3, 3).copy()
        return pos, rot
