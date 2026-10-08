"""Robot descriptions (MuJoCo scene, PD gains, poses, correspondences) and model building.

Each robot is a ``RobotSpec`` in ``ROBOTS``. ``build_model`` rebuilds its MJCF through MjSpec so that
  * foot/floor contacts come from regular collision filtering only (shipped explicit <pair>s
    would duplicate every foot contact),
  * self-collision is active between all collision geoms (the paper relies on the simulator to
    prevent self-penetration), except for body pairs that already overlap in the rest poses,
  * motors become implicit PD position actuators so a whole control step (decimation sub-steps)
    can be captured in a single CUDA graph.

Collision geoms must be spheres or capsules: the environment and the evaluation compute ground
and self-penetration from radius / half-length.

Supported robots:
  * ``g1``: Unitree G1 29 DoF (MJCF from holosoma).
  * ``k1``: Booster K1 22 DoF (MJCF from BoosterRobotics/booster_assets; gains, effort limits and
    default pose from the K1 BeyondMimic config in BoosterRobotics/booster_deploy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from reactor import ASSETS


@dataclass(frozen=True)
class RobotSpec:
    name: str
    scene_xml: Path
    kp: dict[str, float]  # per joint group (see ``joint_group``)
    kd: dict[str, float]
    default_joints: dict[str, float]  # mean of the randomized initial configuration (Sec. 6)
    nominal_joints: dict[str, float]  # T-pose aligned with the SMPL rest pose (Fig. 4)
    # sparse semantic correspondences (robot body, SMPL+H joint index); the first pair is the root
    correspondences: list[tuple[str, int]]
    feet: tuple[str, str]  # (left, right)
    torso: str  # body whose orientation triggers early termination
    foot_clearance: float  # height of the foot body origin above the sole in the nominal pose
    effort: dict[str, float] | None = None  # per joint group; None: the MJCF actuator ctrlrange
    edit_scene: Callable[[mujoco.MjSpec], None] | None = field(default=None, repr=False)

    @property
    def root(self) -> str:
        return self.correspondences[0][0]


def joint_group(joint_name: str) -> str:
    """'left_hip_pitch_joint' -> 'hip_pitch'; K1's 'aaleft_shoulder_pitch_joint' -> 'shoulder_pitch'."""
    name = joint_name.removeprefix("aa").removeprefix("left_").removeprefix("right_")
    return name.removesuffix("_joint")


# ---------------------------------------------------------------------- Unitree G1
# PD gains per joint group (holosoma / BeyondMimic G1 values: 10 Hz natural freq, 2x damping)
_G1_KP = {
    "hip_yaw": 40.179238471, "hip_roll": 99.098427777, "hip_pitch": 40.179238471,
    "knee": 99.098427777, "ankle_pitch": 28.501246196, "ankle_roll": 28.501246196,
    "waist_yaw": 40.179238471, "waist_roll": 28.501246196, "waist_pitch": 28.501246196,
    "shoulder_pitch": 14.250623098, "shoulder_roll": 14.250623098, "shoulder_yaw": 14.250623098,
    "elbow": 14.250623098, "wrist_roll": 14.250623098, "wrist_pitch": 16.778327481,
    "wrist_yaw": 16.778327481,
}
_G1_KD = {
    "hip_yaw": 2.557889765, "hip_roll": 6.308801854, "hip_pitch": 2.557889765,
    "knee": 6.308801854, "ankle_pitch": 1.814445687, "ankle_roll": 1.814445687,
    "waist_yaw": 2.557889765, "waist_roll": 1.814445687, "waist_pitch": 1.814445687,
    "shoulder_pitch": 0.907222843, "shoulder_roll": 0.907222843, "shoulder_yaw": 0.907222843,
    "elbow": 0.907222843, "wrist_roll": 0.907222843, "wrist_pitch": 1.068141502,
    "wrist_yaw": 1.068141502,
}

G1 = RobotSpec(
    name="g1",
    scene_xml=ASSETS / "g1" / "scenes" / "scene_g1_29dof_wbt_plane.xml",
    kp=_G1_KP,
    kd=_G1_KD,
    default_joints={
        "left_hip_pitch_joint": -0.312, "left_knee_joint": 0.669, "left_ankle_pitch_joint": -0.363,
        "right_hip_pitch_joint": -0.312, "right_knee_joint": 0.669, "right_ankle_pitch_joint": -0.363,
        "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 0.6,
        "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2, "right_elbow_joint": 0.6,
    },
    nominal_joints={
        "left_shoulder_roll_joint": np.pi / 2, "right_shoulder_roll_joint": -np.pi / 2,
        "left_elbow_joint": np.pi / 2, "right_elbow_joint": np.pi / 2,
    },
    correspondences=[
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
    ],
    feet=("left_ankle_roll_link", "right_ankle_roll_link"),
    torso="torso_link",
    foot_clearance=0.035,
)


# ---------------------------------------------------------------------- Booster K1
def _k1_scene(spec: mujoco.MjSpec) -> None:
    """Make the shipped K1 MJCF (a bare robot file) behave like the G1 scene.

    The shipped collision geoms are boxes, cylinders and meshes with a very stiff contact
    (solref 0.001, below 2x our 5 ms timestep) on a frictionless (condim 1) ground. We replace
    them by capsules that cover the same volumes and use MuJoCo's default contact parameters.
    """
    spec.option.impratio = 1.0
    spec.option.tolerance = 1e-8
    ground = spec.geom("ground")
    ground.name = "floor"
    ground.condim = 3
    ground.solref, ground.solimp = [0.02, 1.0], [0.9, 0.95, 0.001, 0.5, 2.0]
    for g in list(spec.geoms):
        if g.name != "floor" and (g.contype or g.conaffinity):
            spec.delete(g)

    def capsule(body: str, radius: float, a, b):
        spec.body(body).add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[radius, 0, 0], fromto=[*a, *b],
                                 contype=1, conaffinity=1, group=3, rgba=[0.2, 0.6, 0.2, 0.3],
                                 solref=[0.02, 1.0], solimp=[0.9, 0.95, 0.001, 0.5, 2.0])

    capsule("trunk", 0.065, (0, -0.025, 0.06), (0, 0.025, 0.06))  # chest box 0.12 x 0.16 x 0.2
    capsule("trunk", 0.065, (0, -0.025, 0.14), (0, 0.025, 0.14))
    capsule("trunk", 0.05, (0, -0.04, -0.06), (0, 0.04, -0.06))  # pelvis cylinder
    capsule("aahead_pitch_link", 0.075, (0, 0, 0.04), (0, 0, 0.07))
    for s, y in (("left", 1.0), ("right", -1.0)):
        # hand tip (no body in the shipped model), the wrist correspondence; holosoma's
        # K1 model has the same body as {s}_hand_sphere_link
        spec.body(f"{s}_elbow_yaw_link").add_body(name=f"{s}_hand_tip_link", pos=[0, 0.1 * y, 0])
        capsule(f"{s}_elbow_pitch_link", 0.033, (0, 0.01 * y, 0), (0, 0.12 * y, 0))  # upper arm
        capsule(f"{s}_elbow_yaw_link", 0.03, (0, 0.0, 0), (0, 0.2 * y, 0))  # forearm + hand
        capsule(f"{s}_hip_yaw_link", 0.045, (0, 0, 0), (-0.012, 0, -0.08))  # thigh
        capsule(f"{s}_knee_pitch_link", 0.038, (0, 0, -0.02), (0, 0, -0.18))  # shin
        # sole: the shipped 0.18 x 0.07 box, bottom 3.8 cm below the ankle, as three thin capsules
        for dy in (-0.025, 0.0, 0.025):
            capsule(f"{s}_ankle_roll_link", 0.01, (-0.054, dy, -0.028), (0.106, dy, -0.028))


_K1_ARM = ("head_yaw", "head_pitch", "shoulder_pitch", "shoulder_roll", "elbow_pitch", "elbow_yaw")
_K1_LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch")
_K1_ANKLE = ("ankle_pitch", "ankle_roll")

K1 = RobotSpec(
    name="k1",
    scene_xml=ASSETS / "k1" / "K1_22dof.xml",
    # booster_deploy tasks/beyond_mimic/robots/k1 (K1BeyondMimicControllerCfg)
    kp={**dict.fromkeys(_K1_ARM, 4.0), **dict.fromkeys(_K1_LEG, 80.0), **dict.fromkeys(_K1_ANKLE, 30.0)},
    kd={**dict.fromkeys(_K1_ARM, 1.0), **dict.fromkeys(_K1_LEG, 2.0), **dict.fromkeys(_K1_ANKLE, 2.0)},
    effort={"head_yaw": 6.0, "head_pitch": 6.0, **dict.fromkeys(_K1_ARM[2:], 14.0), "hip_pitch": 30.0,
            "hip_roll": 35.0, "hip_yaw": 20.0, "knee_pitch": 40.0, "ankle_pitch": 20.0, "ankle_roll": 20.0},
    # booster_deploy K1_CFG.default_joint_pos: arms down, legs straight
    default_joints={"left_shoulder_roll_joint": -1.3, "right_shoulder_roll_joint": 1.3},
    # the zero configuration is already a T-pose with straight arms and legs
    nominal_joints={},
    # K1 has no waist and no wrist joints: the trunk stands for the pelvis (no spine pair), the
    # elbow_yaw link is the forearm and the hand tip on it is the wrist (as in holosoma's K1 mapping)
    correspondences=[
        ("trunk", 0),
        ("left_hip_roll_link", 1),
        ("left_knee_pitch_link", 4),
        ("left_ankle_roll_link", 7),
        ("right_hip_roll_link", 2),
        ("right_knee_pitch_link", 5),
        ("right_ankle_roll_link", 8),
        ("left_shoulder_roll_link", 16),
        ("left_elbow_yaw_link", 18),
        ("left_hand_tip_link", 20),
        ("right_shoulder_roll_link", 17),
        ("right_elbow_yaw_link", 19),
        ("right_hand_tip_link", 21),
    ],
    feet=("left_ankle_roll_link", "right_ankle_roll_link"),
    torso="trunk",
    foot_clearance=0.038,
    edit_scene=_k1_scene,
)

ROBOTS = {r.name: r for r in (G1, K1)}


def get_robot(robot: str | RobotSpec) -> RobotSpec:
    if isinstance(robot, RobotSpec):
        return robot
    if robot not in ROBOTS:
        raise ValueError(f"unknown robot '{robot}', available: {', '.join(ROBOTS)}")
    return ROBOTS[robot]


# ---------------------------------------------------------------------- model building
def scene_spec(robot: str | RobotSpec = "g1") -> mujoco.MjSpec:
    """The robot's scene as an MjSpec, with a ground plane named 'floor' and no explicit pairs."""
    r = get_robot(robot)
    spec = mujoco.MjSpec.from_file(str(r.scene_xml))
    for pair in list(spec.pairs):
        spec.delete(pair)
    if r.edit_scene is not None:
        r.edit_scene(spec)
    return spec


def build_model(robot: str | RobotSpec = "g1", timestep: float = 0.005, self_collision: bool = True) -> mujoco.MjModel:
    r = get_robot(robot)
    spec = scene_spec(r)
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
        g = joint_group(act.target)
        lim = r.effort[g] if r.effort is not None else float(act.ctrlrange[1])
        act.set_to_position(kp=r.kp[g], kv=r.kd[g])
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-lim, lim]

    model = spec.compile()
    if self_collision:
        _exclude_rest_overlaps(spec, model, r)
        model = spec.compile()
    return model


def _exclude_rest_overlaps(spec: mujoco.MjSpec, model: mujoco.MjModel, r: RobotSpec) -> None:
    """Exclude body pairs whose collision geoms touch in default or nominal poses."""
    data = mujoco.MjData(model)
    pairs = set()
    for joints in (r.default_joints, r.nominal_joints, {}):
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

    def __init__(self, model: mujoco.MjModel, robot: str | RobotSpec = "g1"):
        r = self.spec = get_robot(robot)
        self.name = r.name
        self.model = model
        self.joint_names = [model.joint(j).name for j in range(1, model.njnt)]
        self.num_joints = len(self.joint_names)
        assert model.nu == self.num_joints
        act_joint = [model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu)]
        assert act_joint == self.joint_names
        self.kp = np.array([r.kp[joint_group(n)] for n in self.joint_names])
        self.effort = model.actuator_forcerange[:, 1].copy()
        self.joint_range = model.jnt_range[1:].copy()
        self.default_q = np.array([r.default_joints.get(n, 0.0) for n in self.joint_names])
        self.nominal_q = np.array([r.nominal_joints.get(n, 0.0) for n in self.joint_names])
        # BeyondMimic-style action scale: 0.25 * effort / stiffness
        self.action_scale = 0.25 * self.effort / self.kp
        self.body_names = [b for b, _ in r.correspondences]
        self.body_ids = np.array([model.body(b).id for b in self.body_names])
        self.smpl_ids = np.array([j for _, j in r.correspondences])
        self.root_body = model.body(r.root).id
        self.torso_pair = self.body_names.index(r.torso)
        self.foot_ids = np.array([model.body(f).id for f in r.feet])
        self.num_pairs = len(r.correspondences)

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
