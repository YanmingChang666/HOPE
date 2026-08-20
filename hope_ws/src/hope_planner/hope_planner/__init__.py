"""HOPE model-based racket planner.

Pure-Python core (constants, estimator, predictor, target planner, contact
model, quaternion utilities, pipeline) with a thin ROS 2 node wrapper in
``node.py``.
"""

# =============================================================================
# 【中文说明】hope_planner 包的公共入口（re-export 常用类/函数，方便 from hope_planner import X）
# -----------------------------------------------------------------------------
#   本包分层：纯 Python 算法核心 + node.py 的薄 ROS 2 封装。此处把核心类统一导出到包顶层，
#   下游/测试可直接 `from hope_planner import HOPEPlanner`，无需记住子模块路径。
#   模块速览：
#     constants  参数/物理常量  ·  ball_state_estimator  位置流→平滑位速
#     ball_trajectory_predictor  前向积分到击球平面  ·  racket_target_planner  反解球拍目标
#     ball_contact  正向接触模型  ·  quaternion_utils  法向→四元数  ·  planner  三级流水线编排
#   __all__ 显式声明对外符号，既是文档也约束 `from hope_planner import *` 的范围。
# =============================================================================

from .ball_contact import orient_normal, predict_paddle_contact
from .ball_state_estimator import BallStateEstimator
from .ball_trajectory_predictor import BallTrajectoryPredictor, StrikeTarget
from .constants import (
    BallPhysics,
    PlannerConfig,
    TableParams,
    load_ball_physics,
    load_paddle_params,
)
from .planner import HOPEPlanner
from .quaternion_utils import normal_to_quaternion
from .racket_target_planner import RacketCommand, RacketTargetPlanner

__all__ = [
    "BallPhysics",
    "PlannerConfig",
    "TableParams",
    "load_ball_physics",
    "load_paddle_params",
    "BallStateEstimator",
    "BallTrajectoryPredictor",
    "StrikeTarget",
    "RacketTargetPlanner",
    "RacketCommand",
    "HOPEPlanner",
    "normal_to_quaternion",
    "orient_normal",
    "predict_paddle_contact",
]
