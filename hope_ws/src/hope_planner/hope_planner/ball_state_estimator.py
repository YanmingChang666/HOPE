"""Ball state estimation.

Fits a low-order polynomial to the most recent position samples and
differentiates it analytically to obtain a smoothed position and velocity.
The buffer is cleared on each detected table bounce so the fit never spans the
velocity discontinuity.
"""

# =============================================================================
# 【中文说明】球状态估计：把“带噪的位置流”变成“平滑的位置+速度”
# -----------------------------------------------------------------------------
#   方法：对最近 fit_window 个位置样本做低阶多项式最小二乘拟合（逐轴 polyfit），
#         对时间解析求导得到速度。相比一阶差分，能显著抑制动捕高频噪声。
#   关键：把时间平移到“最新样本为 0”再拟合（改善数值条件）；取常数项=位置、
#         一次项系数=速度（即 t=0 处的值与导数）。
#   弹跳处理：拟合窗口一旦跨越弹跳（速度方向突变），拟合会失真。因此用三点 z 模式
#         检测桌面弹跳并 reset() 清空缓冲，保证每段拟合都在“同一段连续飞行”内。
#         覆盖两种几何：① 点球触底到 bounce_z_tol；② 中心跟踪球（最低点=球半径）
#         出现的局部 z 极小值（bounce_center_z_max）。
# =============================================================================

from typing import List, Optional, Tuple

import numpy as np

from .constants import PlannerConfig


class BallStateEstimator:
    """Estimate ball position and velocity from a stream of positions.

    Maintains a sliding window of recent position measurements and performs a
    least-squares polynomial fit to extract smoothed position and velocity.

    Bounce detection uses a three-sample z-height pattern to identify a table
    impact and clear the buffer, covering two geometries:

    * a point-ball dip that touches ``bounce_z_tol``;
    * a local z-minimum below ``bounce_center_z_max`` for centre-tracked balls,
      whose minimum height at contact is the ball radius.
    """

    def __init__(self, config: PlannerConfig):
        self.config = config
        self.t_buffer: List[float] = []
        self.p_buffer: List[np.ndarray] = []

        # Three-sample z ring buffer for bounce detection; None suppresses
        # false triggers before enough measurements are collected.
        self._z_hist: List[Optional[float]] = [None, None, None]
        self._bounce_detected: bool = False

    def reset(self) -> None:
        """Clear the estimation buffer (called on bounce detection)."""
        self.t_buffer.clear()
        self.p_buffer.clear()

    def push(self, t: float, p: np.ndarray) -> None:
        """Add a new position measurement.

        Parameters
        ----------
        t : float
            Timestamp in seconds (monotonic).
        p : np.ndarray, shape (3,)
            Ball position [x, y, z] in the play frame.
        """
        # 维护三点 z 环形缓冲（前前、前、当前），用于弹跳检测。
        self._z_hist[0] = self._z_hist[1]
        self._z_hist[1] = self._z_hist[2]
        self._z_hist[2] = p[2]

        self._bounce_detected = False
        z_pp, z_p, z_c = self._z_hist
        tol = self.config.bounce_z_tol
        center_max = getattr(self.config, "bounce_center_z_max", 0.05)
        if z_pp is not None and z_p is not None and z_c is not None:
            # legacy_dip：点球触底——中间帧压到阈值以下、两侧都在阈值以上（V 形触底）。
            legacy_dip = z_pp > tol and z_p <= tol and z_c > tol
            # center_min：中心跟踪球——中间帧是局部极小且贴近桌面（最低点≈球半径）。
            center_min = z_p <= center_max and z_pp > z_p and z_c > z_p
            if legacy_dip or center_min:
                # 检测到弹跳：清空拟合缓冲，避免下一次拟合跨越速度突变而失真。
                self._bounce_detected = True
                self.reset()

        self.t_buffer.append(t)
        self.p_buffer.append(p.copy())

        if len(self.t_buffer) > self.config.fit_window:
            self.t_buffer.pop(0)
            self.p_buffer.pop(0)

    @property
    def bounce_detected(self) -> bool:
        """True if the most recent push() detected a table bounce."""
        return self._bounce_detected

    @property
    def ready(self) -> bool:
        """True once enough samples exist for a stable fit."""
        return len(self.t_buffer) >= 6

    def estimate(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute smoothed ball position and velocity at the latest timestamp.

        Returns
        -------
        p_est : np.ndarray, shape (3,)
            Smoothed position estimate [x, y, z].
        v_est : np.ndarray, shape (3,)
            Velocity estimate [vx, vy, vz] (m/s).
        t_est : float
            Timestamp of the estimate (latest sample time).
        """
        if not self.ready:
            raise RuntimeError(f"Need >= 6 samples, have {len(self.t_buffer)}")

        t_arr = np.array(self.t_buffer)
        p_arr = np.array(self.p_buffer)

        # 把时间平移到“最新样本=0”，改善多项式拟合的数值条件（否则大绝对时间戳会病态）。
        t_ref = t_arr[-1]
        t_norm = t_arr - t_ref

        p_est = np.zeros(3)
        v_est = np.zeros(3)
        for axis in range(3):
            # 逐轴多项式最小二乘拟合；系数按降幂排列。
            coeffs = np.polyfit(t_norm, p_arr[:, axis], deg=self.config.poly_order)
            p_est[axis] = coeffs[-1]   # 常数项 = t_norm=0 处的位置
            v_est[axis] = coeffs[-2]   # 一次项 = t_norm=0 处的一阶导（速度）

        return p_est, v_est, t_ref
