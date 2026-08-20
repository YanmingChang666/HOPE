"""Racket-normal-to-quaternion conversion.

The face normal defines paddle tilt (2 DOF) but not roll about the face axis.
The ``constrain_up`` option aligns the paddle handle (local Y) with world -Z.
"""

# =============================================================================
# 【中文说明】把“球拍面法向”转成朝向四元数 [x, y, z, w]，供下游 IK 使用
# -----------------------------------------------------------------------------
#   规划器只反解出球拍面法向 n（决定拍面倾斜，2 自由度），但绕拍面轴的“自转/roll”
#   自由度是欠约束的。本模块做两件事：
#     ① 求“把参考轴 +x 旋到 n”的最短弧四元数（拍面朝向）；
#     ② 若 constrain_up=True，再叠加一个绕 n 的 roll，使拍柄(局部 Y 轴)对齐世界 -Z
#        (朝下)，从而把最后 1 个自由度也定死，得到唯一确定的姿态。
#   约定：四元数按 [x, y, z, w] 顺序（与 ROS geometry_msgs/Quaternion 一致）。
#   注意：本模块不进 wire 消息（RacketCommand 只发位置/速度）；它是给需要完整姿态的
#         下游逆运动学的“姿态提示”转换工具。
# =============================================================================

import numpy as np


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    # 四元数 [x,y,z,w] → 3x3 旋转矩阵（用于取出局部坐标轴在世界系下的方向）。
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # 四元数乘法 q1⊗q2（[x,y,z,w] 约定）：用于把 roll 修正叠加到基础朝向上。
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def normal_to_quaternion(n_racket: np.ndarray, constrain_up: bool = False) -> np.ndarray:
    """Convert a racket face normal to an orientation quaternion [x, y, z, w]."""
    # 第一步：求“把参考轴 ref=+x 旋到目标法向 n”的最短弧旋转（拍面朝向）。
    n = n_racket / np.linalg.norm(n_racket)
    ref = np.array([1.0, 0.0, 0.0])

    axis = np.cross(ref, n)          # 旋转轴 = ref × n
    sin_angle = np.linalg.norm(axis)  # |ref×n| = sin(夹角)
    cos_angle = np.dot(ref, n)        # ref·n   = cos(夹角)

    if sin_angle < 1e-8:
        # 退化：n 与 ref 几乎共线。同向→单位四元数；反向(180°)→绕 y 轴翻转。
        q = np.array([0.0, 0.0, 0.0, 1.0]) if cos_angle > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    else:
        # 轴角 → 四元数：q = (axis·sin(θ/2), cos(θ/2))。
        axis = axis / sin_angle
        half_angle = np.arctan2(sin_angle, cos_angle) / 2.0
        qw = np.cos(half_angle)
        qxyz = axis * np.sin(half_angle)
        q = np.array([qxyz[0], qxyz[1], qxyz[2], qw])

    if not constrain_up:
        return q   # 不约束 roll：拍面朝向已确定即可返回

    # 第二步（可选）：约束绕拍面轴的 roll——让拍柄(局部 Y)对齐世界 -Z(朝下)。
    # Constrain roll: align paddle handle (local Y) with world -Z.
    R = _quat_to_matrix(q)
    current_y = R @ np.array([0.0, 1.0, 0.0])       # 当前拍柄方向（世界系）
    down = np.array([0.0, 0.0, -1.0])
    desired_y = down - np.dot(down, n) * n          # 把“朝下”投影到拍面内，得到目标拍柄方向
    desired_y_norm = np.linalg.norm(desired_y)
    if desired_y_norm < 1e-6:
        return q   # 拍面近乎水平，"朝下"几乎垂直于拍面，roll 无从定义 → 保持原姿态
    desired_y = desired_y / desired_y_norm

    # 在拍面内求 current_y 到 desired_y 的夹角（以 n 为旋转轴的有符号 roll）。
    cos_roll = np.clip(np.dot(current_y, desired_y), -1.0, 1.0)
    sin_roll = np.dot(np.cross(current_y, desired_y), n)
    roll_half = np.arctan2(sin_roll, cos_roll) / 2.0

    # 构造绕 n 的 roll 四元数，左乘到基础朝向上得到最终姿态。
    q_roll = np.array([
        n[0] * np.sin(roll_half), n[1] * np.sin(roll_half),
        n[2] * np.sin(roll_half), np.cos(roll_half),
    ])
    return _quat_multiply(q_roll, q)
