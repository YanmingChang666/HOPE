"""Physical constants and tuning parameters for the HOPE planner.

Frame convention (matching ``configs/ball_physics.yaml``): +x forward (toward
the opponent), +y left, +z up (right-handed); z = 0 is the table surface and
the world origin is the near-side left table corner, so the table occupies
x in [0, length] and y in [-width, 0]. Ball state is no-spin only:
[x, y, z, vx, vy, vz].

The fitted no-spin ball physics (drag, table/paddle restitution, gravity, ball
geometry, table/net geometry) are read from ``configs/ball_physics.yaml`` when
it is found; see :func:`load_ball_physics`, :func:`load_paddle_params`, and
:func:`load_table_params`. The dataclass defaults below mirror that file and are
used only as a fallback when it is absent, so the pure modules import and run
without any config on disk.
"""

# =============================================================================
# 【中文说明】物理常量与调参参数（整个规划栈的“参数中心”）
# -----------------------------------------------------------------------------
#   本文件定义三组参数数据类，并提供从 configs/ball_physics.yaml 加载它们的函数：
#     · TableParams    球桌/球网几何（长宽高、网位置、网高、网两侧外伸）
#     · BallPhysics    无旋球空气动力学与恢复系数（阻力 k、桌面切/法向恢复、重力、球半径）
#     · PlannerConfig  规划器调参（估计窗口、积分步长、击球平面、目标落点、球拍恢复系数等）
#
#   坐标系约定（与 ball_physics.yaml 一致）：+x 向前(朝对手)、+y 向左、+z 向上(右手系)；
#   z=0 为桌面，世界原点在“近端左角”，故桌面占据 x∈[0,length]、y∈[-width,0]。
#   球态为纯无旋六维：[x, y, z, vx, vy, vz]。
#
#   设计要点：dataclass 里的默认值“镜像”了 YAML 文件，只在磁盘上找不到 YAML 时作兜底，
#   这样纯算法模块无需任何配置文件即可 import 并运行（便于单元测试/离线调试）。
#   加载优先级：显式 path > 自动向上搜索 configs/ball_physics.yaml > dataclass 默认值。
# =============================================================================

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class TableParams:
    """Table / net geometry, expressed in the play frame (table at y in [-width, 0])."""
    # 球桌与球网几何（在比赛坐标系下：桌面占据 y∈[-width, 0]）。

    length: float = 2.74          # m, along +x
    width: float = 1.525          # m, table occupies y in [-width, 0]
    height: float = 0.76          # m, table surface above the floor
    y_max: float = 0.0            # m, table's +y edge in the frame
    net_x: float = 1.37           # m, net plane along x
    net_height: float = 0.1525    # m, net top above the table surface
    net_overhang: float = 0.15    # m, net extends past each table edge in y

    @classmethod
    def from_mapping(cls, data: Dict, y_max: Optional[float] = None) -> "TableParams":
        """Build from a parsed ``ball_physics.yaml`` mapping (missing keys -> defaults)."""
        # 从解析后的 YAML 字典构造；缺失的键一律回落到上面的默认值（容错，绝不抛异常）。
        d = cls()
        if y_max is not None:
            d.y_max = float(y_max)
        geom = data.get("geometry", {}) if isinstance(data, dict) else {}
        geom = geom if isinstance(geom, dict) else {}
        # 优先读顶层 `table`/`net`（当前 ball_physics.yaml 的规范 schema）；
        # 找不到再回退到旧版的 `geometry.table`/`geometry.net` 嵌套结构（向后兼容）。
        # Prefer top-level `table`/`net` (canonical ball_physics.yaml schema);
        # fall back to the older `geometry.table`/`geometry.net` nesting.
        table = data.get("table", geom.get("table", {})) if isinstance(data, dict) else {}
        net = data.get("net", geom.get("net", {})) if isinstance(data, dict) else {}
        if isinstance(table, dict):
            if table.get("length") is not None:
                d.length = float(table["length"])
            if table.get("width") is not None:
                d.width = float(table["width"])
            if table.get("height") is not None:
                d.height = float(table["height"])
        if isinstance(net, dict):
            if net.get("x_position") is not None:
                d.net_x = float(net["x_position"])
            if net.get("height") is not None:
                d.net_height = float(net["height"])
            if net.get("overhang") is not None:
                d.net_overhang = float(net["overhang"])
        return d


@dataclass
class BallPhysics:
    """No-spin aerodynamic and restitution parameters."""
    # 无旋球的空气动力学与恢复系数（预测器与目标规划器共用同一套物理常量）。

    k: float = 0.1261             # 1/m, quadratic drag: a = -k |v| v
    C_h: float = 0.631            # tangential retention at a table bounce (= 1 - a_t)
    C_v: float = 0.9215           # normal restitution at a table bounce
    g: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    radius: float = 0.02          # m, ball radius (40 mm diameter)
    mass: float = 0.0027          # kg, ball mass (drag folds mass in; kept for reference)

    @classmethod
    def from_mapping(cls, data: Dict) -> "BallPhysics":
        """Build from a parsed ``ball_physics.yaml`` mapping (missing keys -> defaults)."""
        # 从 YAML 字典读取，键名兼容多种写法（用 _first 依次尝试候选键）；缺失即用默认值。
        d = cls()
        if not isinstance(data, dict):
            return d
        ball = data.get("ball", {})
        drag = data.get("drag", data.get("flight", {}))       # 阻力段：兼容 drag / flight 两种命名
        contact = data.get("contact", {})                     # 接触段：内含 table / paddle 子表
        table = contact.get("table", {}) if isinstance(contact, dict) else {}

        k = _first(drag, ("k_d", "k", "coefficient"))         # 二次阻力系数（多种别名）
        if k is not None:
            d.k = float(k)
        # 重力只取标量大小，强制写成朝下的 [0,0,-|g|]，避免 YAML 里符号写反导致重力朝上。
        gravity = _first(data, ("gravity",)) or _first(drag, ("g", "gravity"))
        if gravity is not None:
            d.g = np.array([0.0, 0.0, -abs(float(gravity))])
        if isinstance(ball, dict):
            if ball.get("radius") is not None:
                d.radius = float(ball["radius"])
            if ball.get("mass") is not None:
                d.mass = float(ball["mass"])
        # 桌面法向恢复系数 C_v（弹起时竖直速度保留比例）。
        c_v = _first(table, ("restitution", "e_n", "e_eff", "restitution_normal"))
        if c_v is not None:
            d.C_v = float(c_v)
        # 桌面切向保留 C_h：既可直接给 C_h，也可给切向阻尼 a_t 后换算 C_h = 1 - a_t。
        c_h = _first(table, ("restitution_tangential", "tangential_retention"))
        if c_h is not None:
            d.C_h = float(c_h)
        else:
            a_t = _first(table, ("tangential_damping", "a_t"))
            if a_t is not None:
                d.C_h = 1.0 - float(a_t)
        return d


@dataclass
class PlannerConfig:
    """Planner tuning parameters."""
    # 规划器调参：可经由 ROS 参数覆盖，也是 node.py 里 declare_parameter 的默认来源。

    # State estimation —— 状态估计
    poly_order: int = 2           # 多项式拟合阶数（2 阶=位置/速度/加速度）
    fit_window: int = 31          # 速度拟合所用的位置样本数（滑动窗口长度）
    mocap_hz: float = 300.0       # 标称动捕采样率（仅供参考/调参用）

    # Trajectory prediction —— 轨迹预测
    dt_integrate: float = 0.001   # 前向积分步长（秒），越小越精但越慢
    max_predict_time: float = 2.0  # 前向预测时域上限（秒），超时则判无有效击球
    bounce_z_tol: float = 0.005   # 点球模型判弹跳的 z 触底阈值（米）
    bounce_center_z_max: float = 0.05  # 中心跟踪球判弹跳的局部极小高度阈值（米）

    # Racket planning —— 球拍规划
    x_hit: float = 0.0            # 固定虚拟击球平面的 x 坐标（米）
    target_land: np.ndarray = field(
        default_factory=lambda: np.array([2.055, -0.7625, 0.02])
    )                             # fixed landing target (opponent-half centre); z = ball radius,
                                  # the CENTROID height at table contact (same convention as the
                                  # bounce planes everywhere else)
    delta_t_flight: float = 0.5   # 期望的击球后飞行时间（秒），决定所需球拍速度大小
    C_r: float = 0.654            # 球拍法向恢复系数
    racket_radius: float = 0.075  # 米，球拍半径

    # Simplified paddle tangential contact (used by ball_contact.py)
    # 简化的球拍切向接触模型参数（供 ball_contact.py 使用）：
    paddle_a_t: float = 0.52      # 切向阻尼比例
    paddle_b_t: float = 0.0       # 切向项对入射角的耦合系数
    paddle_mu: float = 0.5        # 摩擦锥上限（切向冲量封顶）


def _first(mapping: Dict, keys) -> Optional[float]:
    """Return the first present key's value from a mapping, else None."""
    # 按候选键顺序返回第一个存在且非 None 的值——用于兼容 YAML 里同一含义的多种键名。
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def find_ball_physics_config(start: Optional[Path] = None) -> Optional[Path]:
    """Search upward from ``start`` (or this file) for ``configs/ball_physics.yaml``."""
    # 从 start（或本文件）逐级向上查找 configs/ball_physics.yaml，找到即返回，找不到返回 None。
    base = Path(start) if start is not None else Path(__file__).resolve()
    for parent in [base, *base.parents]:
        candidate = parent / "configs" / "ball_physics.yaml"
        if candidate.is_file():
            return candidate
    return None


def _read_physics_yaml(path: Optional[str] = None) -> Dict:
    """Load the ball-physics YAML as a dict; return {} if missing or unreadable."""
    # 读取物理 YAML 为字典；文件缺失/解析失败一律返回 {}（让上层回落到默认值，绝不崩溃）。
    resolved = Path(path) if path else find_ball_physics_config()
    if not resolved or not Path(resolved).is_file():
        return {}
    try:
        import yaml  # 延迟导入：没装 PyYAML 时纯算法模块仍可 import（只是拿不到 YAML）
        with open(resolved, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_ball_physics(path: Optional[str] = None) -> BallPhysics:
    """Load :class:`BallPhysics` from ``configs/ball_physics.yaml`` (or defaults)."""
    return BallPhysics.from_mapping(_read_physics_yaml(path))


def load_table_params(path: Optional[str] = None, y_max: Optional[float] = None) -> TableParams:
    """Load :class:`TableParams` geometry from the physics YAML (or defaults)."""
    return TableParams.from_mapping(_read_physics_yaml(path), y_max=y_max)


def load_paddle_params(path: Optional[str] = None) -> Dict[str, float]:
    """Load paddle restitution / tangential params from the physics YAML.

    Returns a dict with keys ``C_r``, ``paddle_a_t``, ``paddle_b_t``,
    ``paddle_mu``. Missing entries fall back to the :class:`PlannerConfig`
    defaults so callers can splat the result into a config unconditionally.
    """
    # 从 YAML 的 contact.paddle 段读取球拍恢复/切向参数；缺失项回落到 PlannerConfig 默认值，
    # 因此返回的 dict 一定四键齐全，调用方可无条件 ** 展开进 config。
    defaults = PlannerConfig()
    out = {
        "C_r": defaults.C_r,
        "paddle_a_t": defaults.paddle_a_t,
        "paddle_b_t": defaults.paddle_b_t,
        "paddle_mu": defaults.paddle_mu,
    }
    data = _read_physics_yaml(path)
    contact = data.get("contact", {}) if isinstance(data, dict) else {}
    paddle = contact.get("paddle", {}) if isinstance(contact, dict) else {}
    if isinstance(paddle, dict):
        c_r = _first(paddle, ("restitution", "e_n", "e_eff", "restitution_normal"))
        if c_r is not None:
            out["C_r"] = float(c_r)
        a_t = _first(paddle, ("tangential_damping", "a_t"))
        if a_t is not None:
            out["paddle_a_t"] = float(a_t)
        if paddle.get("b_t") is not None:
            out["paddle_b_t"] = float(paddle["b_t"])
        mu = _first(paddle, ("tangential_cap", "mu"))
        if mu is not None:
            out["paddle_mu"] = float(mu)
    return out
