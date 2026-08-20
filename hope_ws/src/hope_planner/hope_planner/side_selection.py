"""Forehand/backhand selection: a binary split on the predicted lateral crossing.

The implemented convention (documented in ``docs/PLANNER_INTERFACE.md`` and the
planner YAML) is:

    crossing_y <  swing_side_split_y  -> FOREHAND (+1)
    crossing_y >= swing_side_split_y  -> BACKHAND (-1)

i.e. a ball arriving BELOW the split (toward the robot's paddle side, -y) is taken
forehand; a ball at or above the split is taken backhand.

With a nonzero hysteresis band the previous task's side is sticky near the split:

    previous FOREHAND: stay FOREHAND unless crossing_y >  split + hysteresis
    previous BACKHAND: stay BACKHAND unless crossing_y <  split - hysteresis

so a ball arriving near the boundary does not flip the choice between consecutive
rallies. Selection happens once per ``task_id`` and is locked for that task.
"""

# =============================================================================
# 【中文说明】正/反手选择：对“预测的横向落点 y”做二元阈值判定（纯函数，易单测）
# -----------------------------------------------------------------------------
#   约定：crossing_y <  split_y → 正手 FOREHAND(+1)；≥ split_y → 反手 BACKHAND(-1)。
#   迟滞 hysteresis_y>0 时，落点落在 split 附近的迟滞带内会“粘住上一回合的选择”，
#   避免边界球在相邻回合来回翻转（need cross split±hysteresis 才切换）。
#   每个 task_id 只判一次并锁定；由 node.py 调用。
# =============================================================================

from __future__ import annotations

FOREHAND: int = 1
BACKHAND: int = -1


def select_swing_side(
    crossing_y: float,
    split_y: float,
    hysteresis_y: float = 0.0,
    prev_side: int = 0,
) -> int:
    """Return ``FOREHAND`` (+1) or ``BACKHAND`` (-1) for a predicted lateral crossing.

    ``prev_side`` is the side of the previous task (0 = none yet); it only matters
    inside the hysteresis band around ``split_y``.
    """
    hysteresis_y = max(0.0, float(hysteresis_y))
    lo = split_y - hysteresis_y
    hi = split_y + hysteresis_y
    if prev_side == FOREHAND:
        return BACKHAND if crossing_y > hi else FOREHAND
    if prev_side == BACKHAND:
        return FOREHAND if crossing_y < lo else BACKHAND
    return FOREHAND if crossing_y < split_y else BACKHAND
