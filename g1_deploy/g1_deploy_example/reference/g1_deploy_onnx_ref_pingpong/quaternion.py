# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Minimal quaternion helpers used to assemble the 111-D observation.

Convention: quaternions are stored ``(w, x, y, z)`` (Hamilton / scalar-first),
which matches both the MuJoCo free-joint ``qpos`` layout and the training-side
math. All functions operate on plain ``numpy`` arrays and are allocation-cheap so
they can run inside the 50 Hz control loop.

The two rotate primitives follow the standard scalar-first rotation identity:
    quat_rotate(q, v)          rotates a body-frame vector into the world frame
    quat_rotate_inverse(q, v)  rotates a world-frame vector into the body frame
Written from scratch against the public observation contract; no external
reference-frame tables are used.
"""

from __future__ import annotations

import numpy as np

_GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def normalize(q: np.ndarray) -> np.ndarray:
    """Return ``q`` scaled to unit norm; identity if the input is degenerate."""
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate body-frame vector ``v`` into the world frame using quaternion ``q``."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[0]
    xyz = q[1:4]
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(xyz, v) * (2.0 * w)
    c = xyz * (2.0 * float(np.dot(xyz, v)))
    return a + b + c


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector ``v`` into the body frame using quaternion ``q``."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[0]
    xyz = q[1:4]
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(xyz, v) * (2.0 * w)
    c = xyz * (2.0 * float(np.dot(xyz, v)))
    return a - b + c


def projected_gravity_body(q: np.ndarray) -> np.ndarray:
    """Unit gravity direction expressed in the base body frame (IMU term).

    【中文】重力方向投影观测项：把世界系单位重力 (0,0,-1) 用基座朝向的逆旋转到机体系。
    机体水平时约为 (0,0,-1)；机体前倾/侧倾时 x/y 分量变大——等价于“机体倾斜度”。
    真机上这一项等价于 IMU 加速度计静态读出的重力方向。
    """
    return quat_rotate_inverse(q, _GRAVITY_WORLD)


def base_forward_xy(q: np.ndarray) -> np.ndarray:
    """World-XY projection of the base +x axis, renormalized to unit length.

    This is the explicit heading vector the policy uses to resolve world-frame
    goal directions without an absolute yaw reference.

    【中文】基座航向观测项：把机体前向轴 +x 旋到世界系，取其 xy 分量并重新归一化成单位向量。
    这给策略一个“显式航向”，使它无需绝对 yaw 角就能把世界系的目标方向（球拍目标、回中误差）
    解算到自身参照下。
    """
    fwd = quat_rotate(q, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    n = max(float(np.hypot(fwd[0], fwd[1])), 1e-6)
    return np.array([fwd[0] / n, fwd[1] / n], dtype=np.float64)
