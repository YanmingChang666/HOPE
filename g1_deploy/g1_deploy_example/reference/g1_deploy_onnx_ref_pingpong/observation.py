# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Assemble the actor observation (G1: 105-D = 18 + 3*29).

The layout is fixed and must byte-for-byte match the training/export contract.
There is exactly one layout; no normalization is applied (raw observation). The
three joint-width terms scale with the DOF count N; everything else is fixed, so
``OBS_DIM = 3 + 3*N + 15 = 18 + 3*N`` (A3: 111 with N=31; G1: 105 with N=29).

| slice           | term                    | dim | frame / units          |
|-----------------|-------------------------|-----|------------------------|
| [0:3]           | base_ang_vel            | 3   | pelvis body, rad/s     |
| [3:3+N]         | joint_pos               | N   | rad, q - default_q     |
| [3+N:3+2N]      | joint_vel               | N   | rad/s                  |
| [3+2N:3+3N]     | last_action             | N   | applied (prev tick)    |
| ...             | projected_gravity       | 3   | base frame, unit       |
| ...             | base_forward_xy         | 2   | world xy, unit         |
| ...             | fixed_station_error_xy  | 2   | world xy, m            |
| ...             | racket_target_rel_base  | 3   | world, m               |
| ...             | racket_target_vel_w     | 3   | world, m/s             |
| ...             | time_to_strike          | 1   | s                      |
| ...             | swing_side              | 1   | +1 forehand / -1 back  |

``fixed_station_error_xy`` = fixed_station_xy (a startup constant) - current base
xy. It is NOT constant zero: the base drifts across rallies, and this term is the
in-place recentring feedback the policy uses to hold its station.

``last_action`` is the previous tick's APPLIED action. The G1 has no passive neck,
so no columns are zeroed (HEAD_INDICES is empty); the applied action equals the
actor's raw output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quaternion as quat
from .joint_order import NUM_JOINTS

OBS_DIM: int = 18 + 3 * NUM_JOINTS  # A3: 111, G1: 105
_ACTION_DIM: int = NUM_JOINTS


@dataclass
class RobotState:
    """Proprioceptive robot state consumed by the observation builder.

    All world poses are expressed in the same frame the planner uses for the
    racket target. On hardware this is the operator/table frame; in the reference
    MuJoCo sim it is the simulator world frame.
    """

    base_pos_w: np.ndarray      # (3,) pelvis position, world m
    base_quat_w: np.ndarray     # (4,) pelvis orientation, (w, x, y, z)
    base_ang_vel_b: np.ndarray  # (3,) pelvis gyro, body frame rad/s
    q: np.ndarray               # (31,) joint positions, joint order
    qd: np.ndarray              # (31,) joint velocities, joint order


@dataclass
class ObsTarget:
    """The strike goal fed into the observation for the current tick."""

    pos_w: np.ndarray   # (3,) racket target position, world m
    vel_w: np.ndarray   # (3,) racket target velocity, world m/s
    time_to_strike: float
    swing_side: float   # +1 forehand / -1 backhand


def build_observation(
    state: RobotState,
    target: ObsTarget,
    last_action: np.ndarray,
    default_q: np.ndarray,
    fixed_station_xy: np.ndarray,
) -> np.ndarray:
    """Return the 111-D observation as a contiguous ``float32`` vector.

    【中文说明 —— 观测（observation）的唯一组装点】
    这个函数就是 sim2sim（MuJoCo 评估）和 sim2real（真机部署）共用的 *同一份*
    观测拼装代码。策略网络（导出的 ONNX）在每个 50Hz 控制周期都吃这里返回的这个
    向量。G1 的观测维度 = 18 + 3*29 = 105。对应关系必须和训练/导出时逐字节一致，
    否则策略吃到的数字含义就错了。

    5 个入参的来源：
      * ``state``            —— 机器人本体感知状态（RobotState）。仿真里由
                                MuJoCo 读出（见 sim_bridge.py / mujoco_pingpong_scene.py
                                的 read_state / read_robot_state）；真机上由 IMU + 关节
                                编码器读出。包含：基座位置/朝向四元数、基座角速度(陀螺仪)、
                                关节角 q、关节角速度 qd。
      * ``target``           —— 本周期要击打的目标（ObsTarget），由挥拍状态机
                                lifecycle.update() 依据规划器发来的 RacketCommand 生成：
                                球拍目标位置/速度、距击球时间、正/反手。
      * ``last_action``      —— 上一周期“实际下发”的动作（29 维）。G1 无被动颈部，
                                所以实际下发动作 == 策略原始输出。
      * ``default_q``        —— 默认站姿关节角（29 维），用于把 q 变成相对量 q-default_q。
      * ``fixed_station_xy`` —— 启动时刻锁定的站位 xy（常量），用于计算回中误差。
    这里不做任何归一化（raw observation）。
    """
    q = np.asarray(state.q, dtype=np.float64)                    # 关节角（rad），关节顺序见 joint_order.py
    qd = np.asarray(state.qd, dtype=np.float64)                  # 关节角速度（rad/s）
    default_q = np.asarray(default_q, dtype=np.float64)          # 默认站姿关节角（rad）
    last_action = np.asarray(last_action, dtype=np.float64)      # 上一周期实际下发的动作
    base_pos = np.asarray(state.base_pos_w, dtype=np.float64)    # 基座(骨盆)世界坐标位置（m）
    base_quat = quat.normalize(state.base_quat_w)               # 基座朝向四元数(w,x,y,z)，先归一化到单位长度

    if q.shape[0] != NUM_JOINTS or qd.shape[0] != NUM_JOINTS:
        raise ValueError(f"joint vectors must be length {NUM_JOINTS}")
    if last_action.shape[0] != _ACTION_DIM:
        raise ValueError(f"last_action must be length {_ACTION_DIM}")

    # 按固定顺序把 105 维观测逐段写入。o 是写指针，逐段推进；末尾 assert 校验总长。
    # 每一段的“含义 / 坐标系 / 单位 / 来源”见下方逐行中文注释。
    obs = np.empty(OBS_DIM, dtype=np.float64)
    o = 0
    # [0:3] 基座角速度：陀螺仪读数，机体系(pelvis)，rad/s。来自 state.base_ang_vel_b。
    obs[o:o + 3] = state.base_ang_vel_b;                       o += 3          # base_ang_vel
    # [3:3+N] 关节角相对量：q - 默认站姿，rad。这样 0 就代表“站在默认姿态”。
    obs[o:o + NUM_JOINTS] = q - default_q;                     o += NUM_JOINTS  # joint_pos
    # [3+N:3+2N] 关节角速度：rad/s，直接来自编码器/仿真。
    obs[o:o + NUM_JOINTS] = qd;                                o += NUM_JOINTS  # joint_vel
    # [3+2N:3+3N] 上一周期实际下发动作：与训练时反馈给策略的 last_action 一致。
    obs[o:o + NUM_JOINTS] = last_action;                       o += NUM_JOINTS  # last_action
    # 重力方向投影：把世界系重力(0,0,-1)旋到机体系，单位向量。等价于告诉策略“机体倾斜了多少”。
    obs[o:o + 3] = quat.projected_gravity_body(base_quat);     o += 3          # projected_gravity
    # 基座前向在世界 xy 平面的投影(单位向量)：显式航向，让策略无需绝对 yaw 就能解算世界系目标方向。
    obs[o:o + 2] = quat.base_forward_xy(base_quat);            o += 2          # base_forward_xy
    # 回中误差 xy：启动站位常量 - 当前基座 xy（世界系，m）。不是恒为 0——基座会随回合漂移，
    # 这是策略用来“原地回中”的反馈项。
    obs[o:o + 2] = np.asarray(fixed_station_xy)[:2] - base_pos[:2]; o += 2     # fixed_station_error_xy
    # 球拍目标相对基座的位置：目标世界位置 - 基座世界位置（世界系，m）。目标来自规划器/lifecycle。
    obs[o:o + 3] = np.asarray(target.pos_w) - base_pos;        o += 3          # racket_target_rel_base
    # 球拍目标速度：世界系，m/s。策略据此把球打向对方半台。
    obs[o:o + 3] = np.asarray(target.vel_w);                   o += 3          # racket_target_vel_w
    # 距击球时间：秒。由 lifecycle 参考时钟每周期减 dt，击球时穿过 0，随后为负（跟随挥拍收尾）。
    obs[o] = float(target.time_to_strike);                     o += 1          # time_to_strike
    # 挥拍方向：+1 正手 / -1 反手。整个 task 内锁定。
    obs[o] = float(target.swing_side);                         o += 1          # swing_side
    assert o == OBS_DIM, o                                     # 校验：正好写满 105 维
    return obs.astype(np.float32)                              # 导出为连续 float32 供 ONNX 推理
