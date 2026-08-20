"""Top-level HOPE planner pipeline.

Call :meth:`HOPEPlanner.update` with each ball position at the motion-capture
sample rate; it estimates the ball state, predicts the no-spin trajectory to
the fixed hitting plane, and returns the desired racket command (or None when
there is no usable strike yet).
"""

# =============================================================================
# 【中文说明】三级流水线编排器（纯 Python，无 ROS 依赖，便于单测）
# -----------------------------------------------------------------------------
#   update(t, p_ball) 每来一帧球位调用一次，内部依次跑：
#     ① BallStateEstimator     由位置流拟合出平滑的球位/球速（含弹跳分段）
#     ② BallTrajectoryPredictor 从当前球态前向积分到击球平面 x_hit，得到落点球态
#     ③ RacketTargetPlanner     反解“要把球打到 target_land 所需的球拍速度/法向”
#
#   返回 None 的三种情形（都属于“本帧没有可执行的击球”）：
#     · 样本不足（estimator 未 ready）；
#     · 球在远离机器人（vx≥0，不是来球）；
#     · 预测无有效击球平面穿越（strike.valid==False，如死球/擦网出界）。
#
#   ball_incoming 语义：一旦拿到速度即为 True/False（vx<0=朝机器人来），供 ROS 节点
#   判断“回合是否结束”；估计器还没速度时为 None。
# =============================================================================

from typing import Optional

import numpy as np

from .ball_state_estimator import BallStateEstimator
from .ball_trajectory_predictor import BallTrajectoryPredictor, StrikeTarget
from .constants import BallPhysics, PlannerConfig, TableParams
from .racket_target_planner import RacketCommand, RacketTargetPlanner


class HOPEPlanner:
    """Ball estimation -> trajectory prediction -> racket target planning."""

    def __init__(
        self,
        physics: Optional[BallPhysics] = None,
        config: Optional[PlannerConfig] = None,
        table: Optional[TableParams] = None,
    ):
        self.physics = physics or BallPhysics()
        self.config = config or PlannerConfig()
        self.table = table or TableParams()

        self.estimator = BallStateEstimator(self.config)
        self.predictor = BallTrajectoryPredictor(self.physics, self.config, self.table)
        self.target_planner = RacketTargetPlanner(self.physics, self.config, self.table)

        self._latest_command: Optional[RacketCommand] = None
        self._latest_strike: Optional[StrikeTarget] = None
        self._latest_t: Optional[float] = None
        self._incoming: Optional[bool] = None

    def update(self, t: float, p_ball: np.ndarray) -> Optional[RacketCommand]:
        """Process a new ball position measurement.

        Returns a :class:`RacketCommand` when the ball is incoming and predicted
        to cross the hitting plane, otherwise None.
        """
        self.estimator.push(t, p_ball)

        # ① 样本不足：还拟合不出稳定速度 → 本帧不出指令（incoming 也未知）。
        if not self.estimator.ready:
            self._latest_command = None
            self._latest_strike = None
            self._incoming = None
            return None

        p_est, v_est, t_est = self.estimator.estimate()
        self._latest_t = t_est
        self._incoming = bool(v_est[0] < 0.0)   # vx<0 = 朝机器人飞来

        # ② 只对“朝机器人来的球”(vx<0) 规划；vx≥0 表示球在远离/已被击出。
        if v_est[0] >= 0:
            self._latest_command = None
            self._latest_strike = None
            return None

        # ③ 前向积分预测到击球平面；无有效穿越（死球/擦网/出界）→ 不出指令。
        strike = self.predictor.predict(p_est, v_est, t_est)
        if not strike.valid:
            self._latest_command = None
            self._latest_strike = None
            return None

        # ④ 有有效击球点 → 反解球拍目标，缓存并返回。
        self._latest_strike = strike
        self._latest_command = self.target_planner.plan(strike)
        return self._latest_command

    @property
    def racket_command(self) -> Optional[RacketCommand]:
        return self._latest_command

    @property
    def ball_incoming(self) -> Optional[bool]:
        """True/False once a velocity is available (vx < 0 = toward the robot); else None."""
        return self._incoming

    @property
    def strike_target(self) -> Optional[StrikeTarget]:
        """Latest predicted ball state at the hitting plane (or None)."""
        return self._latest_strike

    @property
    def time_to_strike(self) -> Optional[float]:
        """Seconds remaining until the predicted strike (positive, decreasing)."""
        if self._latest_strike is None or self._latest_t is None:
            return None
        return self._latest_strike.t_strike - self._latest_t
