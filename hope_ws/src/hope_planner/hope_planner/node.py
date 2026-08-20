"""ROS 2 node for the HOPE no-spin racket planner.

Subscribes to the mocap ball stream (``geometry_msgs/PoseArray`` on the
configured poses topic, ball at ``ball_pose_index``), estimates the ball
position/velocity, predicts the no-spin trajectory to a fixed strike plane,
selects forehand/backhand by splitting the predicted lateral (y) position, and
publishes the typed ``hope_msgs/RacketCommand``.

Lifecycle: each new incoming ball gets a new ``task_id``; pre-strike updates
keep that id and increase ``task_revision``; ``swing_side`` is chosen once per
task and locked within it. The first sample seeds position and subsequent
samples enable the velocity fit, after which commands are published directly —
there is no readiness/validity/failure state.
"""

# =============================================================================
# 【中文说明】hope_planner 的 ROS 2 入口节点（整个规划栈的“对外接口层”）
# -----------------------------------------------------------------------------
# 作用：把“动捕球流”翻译成“球拍目标指令”。它本身不含算法，只做 ROS 收发 + 任务
#       生命周期管理，真正的估计/预测/规划都委托给纯 Python 的 HOPEPlanner。
#
# 数据流：
#   /poses (geometry_msgs/PoseArray, 动捕/VRPN 发布, ~180-300Hz)
#        └─► _poses_cb ──► HOPEPlanner.update(t, p_ball)
#                              ├─ BallStateEstimator   估计球的平滑位置/速度
#                              ├─ BallTrajectoryPredictor 预测到击球平面 x_hit 的落点
#                              └─ RacketTargetPlanner   反解球拍目标位姿/速度
#        └─► RacketCommand (hope_msgs/RacketCommand) 发布到 /racket/command
#
# 任务生命周期（关键设计）：
#   · 每来一个“新的来球”→ 新 task_id（+1），task_revision 归零；
#   · 击球前每次重算 → 同一 task_id、task_revision 递增（细化预测）；
#   · swing_side（正手/反手）在一个 task_id 内只判一次并锁定，避免同一回合抖动切换；
#   · 球被击出/远离 → 结束当前 task（下一个来球开新 task_id）。
#
# QoS 设计：
#   · 球流用 BEST_EFFORT + depth=1（只关心最新帧，丢旧帧不重传，低延迟）；
#   · 命令用 RELIABLE + depth=10（下游执行器不能丢指令，需可靠送达）。
#
# 坐标系：全部为世界系（+x 向前/朝对手，+y 向左，+z 向上；米、秒）。
# =============================================================================

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from hope_msgs.msg import RacketCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .constants import PlannerConfig, load_ball_physics, load_paddle_params, load_table_params
from .planner import HOPEPlanner
from .side_selection import select_swing_side

_TASK_ID_WRAP = 1 << 64
_REVISION_WRAP = 1 << 32


class HOPEPlannerNode(Node):
    """ROS 2 wrapper around :class:`HOPEPlanner`."""

    def __init__(self):
        super().__init__("hope_planner")

        # --- Topics / frames ---
        self.declare_parameter("poses_topic", "/poses")
        self.declare_parameter("command_topic", "/racket/command")
        self.declare_parameter("frame_id", "world")
        # Which slot in the PoseArray is the ball (PoseArray carries no names).
        self.declare_parameter("ball_pose_index", 0)

        # --- Planner geometry / tuning ---
        self.declare_parameter("x_hit", 0.0)              # fixed strike-plane x (m)
        self.declare_parameter("swing_side_split_y", -0.7625)   # FH/BH split on predicted y (m)
        self.declare_parameter("swing_side_hysteresis_y", 0.0)  # optional band around the split (m)
        self.declare_parameter("target_land_x", 2.055)   # fixed landing target x (m)
        self.declare_parameter("target_land_y", -0.7625)  # fixed landing target y (m)
        self.declare_parameter("delta_t_flight", 0.5)     # desired post-strike flight time (s)
        self.declare_parameter("max_predict_time", 2.0)   # prediction horizon (s)
        # Push every mocap sample into the estimator; run the predict+plan solve
        # at most every solve_period_s (<= 50 Hz). 0.0 = solve on every sample.
        self.declare_parameter("solve_period_s", 0.02)
        # Table's +y edge in the play frame (table occupies y in [y_max - width, y_max]).
        self.declare_parameter("table_y_max", 0.0)
        # Optional explicit path to configs/ball_physics.yaml ("" = auto-discover).
        self.declare_parameter("ball_physics_path", "")

        self._ball_index = int(self.get_parameter("ball_pose_index").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._split_y = float(self.get_parameter("swing_side_split_y").value)
        self._hysteresis_y = max(0.0, float(self.get_parameter("swing_side_hysteresis_y").value))
        self._solve_period = float(self.get_parameter("solve_period_s").value)

        physics_path = str(self.get_parameter("ball_physics_path").value) or None
        physics = load_ball_physics(physics_path)
        paddle = load_paddle_params(physics_path)
        table = load_table_params(physics_path, y_max=float(self.get_parameter("table_y_max").value))
        config = PlannerConfig(
            x_hit=float(self.get_parameter("x_hit").value),
            # Landing target z = ball radius: the outgoing arc is solved for the ball
            # CENTROID reaching table contact, matching the bounce-plane convention.
            target_land=np.array([
                float(self.get_parameter("target_land_x").value),
                float(self.get_parameter("target_land_y").value),
                physics.radius,
            ]),
            delta_t_flight=float(self.get_parameter("delta_t_flight").value),
            max_predict_time=float(self.get_parameter("max_predict_time").value),
            C_r=paddle["C_r"],
            paddle_a_t=paddle["paddle_a_t"],
            paddle_b_t=paddle["paddle_b_t"],
            paddle_mu=paddle["paddle_mu"],
        )
        self.planner = HOPEPlanner(physics=physics, config=config, table=table)

        # --- Task lifecycle state ---
        self._task_id = 0
        self._task_revision = 0
        self._task_active = False
        self._locked_side = RacketCommand.FOREHAND
        self._prev_side = 0            # side of the previous task (for hysteresis); 0 = none
        self._last_solve_t = None

        mocap_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            PoseArray, str(self.get_parameter("poses_topic").value), self._poses_cb, mocap_qos)
        self.cmd_pub = self.create_publisher(
            RacketCommand, str(self.get_parameter("command_topic").value), command_qos)

        self.get_logger().info(
            f"HOPE planner started: x_hit={config.x_hit:.3f} m, "
            f"landing={config.target_land[:2]}, split_y={self._split_y:.3f} m, "
            f"solve_period={self._solve_period:.3f} s, ball_pose_index={self._ball_index}")

    def _select_side(self, intercept_y: float) -> int:
        """Binary forehand/backhand split on the predicted lateral y (optional hysteresis).

        y below the split -> FOREHAND, at/above -> BACKHAND (the convention in
        docs/PLANNER_INTERFACE.md). Delegates to the pure
        :func:`~hope_planner.side_selection.select_swing_side`, whose boundary
        behaviour is pinned by test/test_side_selection.py. The ROS message
        constants match the pure module's (+1 / -1) by definition of the msg.
        """
        return select_swing_side(
            float(intercept_y), self._split_y, self._hysteresis_y, self._prev_side
        )

    def _poses_cb(self, msg: PoseArray) -> None:
        # 球流回调（每帧动捕都会触发）。PoseArray 里没有名字，用 ball_pose_index 取球所在槽位。
        if len(msg.poses) <= self._ball_index:
            return
        # 时间戳来自消息头（秒+纳秒），供速度拟合与 time_to_strike 计算使用。
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pose = msg.poses[self._ball_index]
        p_ball = np.array([pose.position.x, pose.position.y, pose.position.z])

        # 每一帧都喂给估计器（不丢帧，保证速度拟合窗口密度），但“预测+规划”这套较重的
        # 求解按 solve_period_s 限频（≤50Hz）以省算力：限频窗口内只 push、直接返回。
        if (self._solve_period > 0.0 and self._last_solve_t is not None
                and 0.0 <= (t - self._last_solve_t) < self._solve_period):
            self.planner.estimator.push(t, p_ball)
            return
        self._last_solve_t = t

        # A degenerate mocap frame must degrade to "no command", never kill the node.
        try:
            cmd = self.planner.update(t, p_ball)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            self.get_logger().warning(
                f"planner solve skipped ({type(exc).__name__}: {exc}); check the mocap feed "
                "(units, frame, outliers)", throttle_duration_sec=2.0)
            return

        if cmd is None:
            # 无可用击球目标。若明确判定“球在远离”（ball_incoming==False），
            # 说明已击出/回合结束 → 关闭当前 task，下一个来球会开新的 task_id。
            if self.planner.ball_incoming is False:
                self._task_active = False
            return

        if not self._task_active:
            # 新回合的第一条有效指令：开新 task_id，选定并锁定正/反手（本回合内不再改）。
            self._task_id = (self._task_id + 1) % _TASK_ID_WRAP
            self._task_revision = 0
            self._locked_side = self._select_side(float(cmd.p_intercept[1]))
            self._prev_side = self._locked_side
            self._task_active = True
        else:
            # 同一回合的后续细化：task_id 不变，仅递增 revision（下游据此识别是同一目标的更新）。
            self._task_revision = (self._task_revision + 1) % _REVISION_WRAP

        self._publish(cmd, msg.header)

    def _publish(self, cmd, header) -> None:
        out = RacketCommand()
        out.header = header
        out.header.frame_id = self._frame_id
        out.task_id = self._task_id
        out.task_revision = self._task_revision
        out.swing_side = self._locked_side
        out.position.x = float(cmd.p_intercept[0])
        out.position.y = float(cmd.p_intercept[1])
        out.position.z = float(cmd.p_intercept[2])
        out.velocity.x = float(cmd.v_racket[0])
        out.velocity.y = float(cmd.v_racket[1])
        out.velocity.z = float(cmd.v_racket[2])
        tts = self.planner.time_to_strike
        out.time_to_strike = float(tts) if tts is not None else 0.0
        self.cmd_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HOPEPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
