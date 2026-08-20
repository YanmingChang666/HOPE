# Racket target command published by the HOPE planner.
#
# One task_id is created per incoming ball. Pre-strike updates keep the same
# task_id and increase task_revision. swing_side is chosen once per task_id and
# stays fixed within it. All fields refer to the world frame (+x forward,
# +y left, +z up; metres, seconds).
#
# 【中文说明】HOPE 规划器发布的球拍目标指令（发布到 /racket/command）。
#   任务生命周期：每来一个新球 → 新 task_id；击球前的多次细化沿用同一 task_id、递增
#   task_revision；swing_side(正/反手) 在一个 task_id 内只判一次并锁定。
#   所有字段均为世界系（+x 向前/朝对手、+y 向左、+z 向上；单位 米、秒）。
std_msgs/Header header

# Identifies the incoming ball this command belongs to (new ball -> new id).
# 本条指令归属的来球编号（新球→新 id），下游据此区分不同回合。
uint64 task_id
# Monotonically increasing within a task_id as the pre-strike plan is refined.
# 同一 task_id 内随击球前预测不断细化而单调递增（同一目标的“第几版更新”）。
uint32 task_revision

# swing_side values. —— 正/反手取值常量
int8 FOREHAND=1
int8 BACKHAND=-1
# Selected swing side for this task (FOREHAND or BACKHAND). Formal side channel.
# 本任务选定的挥拍侧(正手/反手)，一经锁定回合内不变。
int8 swing_side

# Target racket position at the strike, world frame (m).
# 击球时刻的球拍目标位置（世界系，米）。
geometry_msgs/Point position
# Target racket velocity at the strike, world frame (m/s).
# 击球时刻的球拍目标速度（世界系，米/秒）。
geometry_msgs/Vector3 velocity
# Remaining time until the strike (s).
# 距离击球还剩多少时间（秒），随预测更新递减。
float64 time_to_strike
