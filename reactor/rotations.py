"""Batched rotation utilities (torch). Quaternions are (w, x, y, z)."""

from __future__ import annotations

import torch


def skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack(
        [z, -v[..., 2], v[..., 1], v[..., 2], z, -v[..., 0], -v[..., 1], v[..., 0], z], dim=-1
    ).reshape(*v.shape, 3)


def exp_map(r: torch.Tensor) -> torch.Tensor:
    """Rotation vector -> rotation matrix (Rodrigues), differentiable at 0."""
    theta2 = (r * r).sum(-1, keepdim=True).unsqueeze(-1)
    theta = torch.sqrt(theta2 + 1e-12)
    K = skew(r)
    small = theta2 < 1e-8
    a = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / (theta2 + 1e-12))
    eye = torch.eye(3, device=r.device, dtype=r.dtype).expand(K.shape)
    return eye + a * K + b * (K @ K)


def log_map(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> rotation vector. Robust near 0 and pi via the quaternion."""
    return quat_to_rotvec(mat_to_quat(R))


def mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    q = torch.stack(
        [
            1.0 + tr,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    )
    # pick the numerically best of the four candidate formulas per element
    idx = q.argmax(-1, keepdim=True)
    r21, r12 = R[..., 2, 1], R[..., 1, 2]
    r02, r20 = R[..., 0, 2], R[..., 2, 0]
    r10, r01 = R[..., 1, 0], R[..., 0, 1]
    c0 = torch.stack([1.0 + tr, r21 - r12, r02 - r20, r10 - r01], -1)
    c1 = torch.stack([r21 - r12, 1.0 + m00 - m11 - m22, r01 + r10, r02 + r20], -1)
    c2 = torch.stack([r02 - r20, r01 + r10, 1.0 - m00 + m11 - m22, r12 + r21], -1)
    c3 = torch.stack([r10 - r01, r02 + r20, r12 + r21, 1.0 - m00 - m11 + m22], -1)
    cands = torch.stack([c0, c1, c2, c3], -2)
    out = torch.gather(cands, -2, idx.unsqueeze(-1).expand(*idx.shape[:-1], 1, 4)).squeeze(-2)
    out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return out * torch.where(out[..., :1] < 0, -1.0, 1.0)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    s = v.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, q[..., :1])
    scale = torch.where(s < 1e-7, 2.0 / q[..., :1].clamp_min(1e-7), angle / s.clamp_min(1e-12))
    return v * scale


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u = q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(u, t, dim=-1)


def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_conj(q), v)


def mat_to_6d(R: torch.Tensor) -> torch.Tensor:
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def geodesic_sq(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """||Log(Ra^T Rb)||^2 computed from the trace (cheap, used in rewards)."""
    tr = (Ra * Rb).sum((-1, -2))
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos) ** 2


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zeros = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), zeros, zeros, torch.sin(yaw / 2)], -1)
