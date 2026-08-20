"""No-spin ball-paddle contact model.

Given the incoming ball velocity, the racket contact-point velocity, and the
racket face normal, predict the outgoing ball velocity with the same
simplified impulse model as ``configs/ball_physics.yaml``:

* normal restitution      v_n_out = -e_n v_n_in;
* tangential damping       s = clip((a_t + b_t cos_theta) |u_t|, 0,
                               mu (1 + e_n) |u_n|),
  v_t_out = v_t_in - s * unit(u_t),  cos_theta = |u_n| / |u|.

Parameters come from :class:`PlannerConfig` (``C_r`` normal restitution,
``paddle_a_t``, ``paddle_b_t``, ``paddle_mu``). No spin is modelled.
"""

# =============================================================================
# 【中文说明】无旋球-球拍接触模型：由“入射球速+球拍面速度+球拍法向”预测出射球速
# -----------------------------------------------------------------------------
#   与 configs/ball_physics.yaml 采用同一套简化冲量模型（不建模自旋）：
#     · 法向：恢复系数模型  v_n_out = -e_n · v_n_in（在球拍参考系内，e_n = C_r）；
#     · 切向：阻尼 + 摩擦锥封顶
#         s = clip((a_t + b_t·cosθ)·|u_t|,  0,  μ(1+e_n)|u_n|)，
#         v_t_out = v_t_in - s·unit(u_t)，其中 cosθ = |u_n|/|u|（入射角度量）。
#   这里 u = v_ball - v_racket 是“球相对球拍表面”的相对速度，法向/切向都在其上分解。
#
#   用途：正向预测“这样挥拍会把球打成什么样”，可用于校验 racket_target_planner 的反解，
#         或在仿真中把球拍动作映射为出球。它是 predict_paddle_contact 的正问题版本。
#   注意符号约定：orient_normal 会把法向翻正，使其指向“入射球来的一侧”，下面的冲量公式
#         正是基于这个朝向推导的——传入的法向可任意正负/长度，内部会归一并定向。
# =============================================================================

import numpy as np

from .constants import BallPhysics, PlannerConfig

_EPS = 1e-9


def orient_normal(n: np.ndarray, v_minus: np.ndarray, v_r: np.ndarray) -> np.ndarray:
    """Orient ``n`` so the incoming relative velocity approaches the face.

    Normalize, then flip when ``(v_minus - v_r) . n > 0`` so the returned normal
    points from the face toward the incoming-ball side (the sign convention the
    impulse equations below assume).
    """
    # 先归一化法向；再看“球相对球拍的速度”在法向上的投影：若为正说明法向与来球同向，
    # 需翻转，使法向指向来球一侧（下面冲量公式假设的符号约定）。
    n = np.asarray(n, dtype=float)
    n = n / (np.linalg.norm(n) + _EPS)
    if float(np.dot(np.asarray(v_minus, dtype=float) - np.asarray(v_r, dtype=float), n)) > 0.0:
        return -n
    return n


def predict_paddle_contact(
    v_minus: np.ndarray,
    v_r: np.ndarray,
    n: np.ndarray,
    physics: BallPhysics,
    config: PlannerConfig,
) -> np.ndarray:
    """Outgoing ball velocity ``v_plus`` from the no-spin impulse model.

    Parameters
    ----------
    v_minus : incoming ball velocity (world frame, m/s).
    v_r : racket contact-point velocity (world frame, m/s).
    n : racket face normal (any sign/length; re-oriented internally).
    physics, config : contact constants.
    """
    v_minus = np.asarray(v_minus, dtype=float)
    v_r = np.asarray(v_r, dtype=float)

    n = orient_normal(n, v_minus, v_r)   # 归一并定向法向（指向来球一侧）

    # 球相对球拍表面的相对速度 u，并分解为法向分量 u_n 与切向分量 u_t。
    # Relative velocity of the ball w.r.t. the racket surface.
    u = v_minus - v_r
    u_n_signed = float(np.dot(u, n))     # 法向有符号分量
    u_t_vec = u - u_n_signed * n         # 切向向量
    u_t_mag = float(np.linalg.norm(u_t_vec))
    u_n_abs = abs(u_n_signed)
    u_mag = float(np.linalg.norm(u))

    e_n = config.C_r
    cos_theta = u_n_abs / (u_mag + _EPS)                              # 入射角余弦 = |u_n|/|u|
    raw = (config.paddle_a_t + config.paddle_b_t * cos_theta) * u_t_mag  # 切向冲量原始值
    cap = config.paddle_mu * (1.0 + e_n) * u_n_abs                    # 摩擦锥上限（不能超过）
    s = min(max(raw, 0.0), cap)                                      # 夹到 [0, cap]

    # 切向：沿 -u_t 方向施加大小为 s 的速度改变（切向几乎为零时不动，避免除零）。
    if u_t_mag > _EPS:
        delta_v_t = -s * (u_t_vec / (u_t_mag + _EPS))
    else:
        delta_v_t = np.zeros(3)

    # 法向：恢复系数模型的速度改变量 Δv_n = -(1+e_n)·u_n·n。
    delta_v_n = -(1.0 + e_n) * u_n_signed * n
    return v_minus + delta_v_n + delta_v_t   # 出射球速 = 入射 + 法向改变 + 切向改变
