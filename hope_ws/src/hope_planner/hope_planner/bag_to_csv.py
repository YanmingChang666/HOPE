"""Convert a rosbag2 topic stream into HOPE calibration CSV (t,x,y,z).

This helper is intentionally small and practical:

- Input: a rosbag2 directory recorded from the standard bringup's ``/poses``
  (``geometry_msgs/PoseArray``, ball at ``--ball-index``, default 0 — so
  calibration capture is simply ``ros2 bag record /poses``); a legacy
  ``geometry_msgs/PointStamped`` topic (e.g. ``/ball/point``) is also accepted.
- Output: one CSV with columns `t,x,y,z`
- Time source: message header stamp if present, otherwise bag receive time
- Storage plugin: auto-detect `mcap` vs `sqlite3`
"""

# =============================================================================
# 【中文说明】把 rosbag2 里的球流导出成 HOPE 标定用 CSV（列：t, x, y, z）
# -----------------------------------------------------------------------------
#   用途：录制球轨迹用于离线标定/拟合球物理参数。工作流就是 `ros2 bag record /poses`，
#         再用本脚本把 bag 转成简单 CSV 喂给拟合工具。
#   输入话题两种都支持：
#     · geometry_msgs/PoseArray（标准 /poses 流，球在 --ball-index 槽位，默认 0）；
#     · 旧版 geometry_msgs/PointStamped（如 /ball/point）。
#   时间来源：优先用消息头时间戳（非零时）；否则退回 bag 的接收时间。
#   存储后端：自动识别 mcap 或 sqlite3（先看 metadata.yaml，再看 *.mcap/*.db3 文件）。
#   健壮性：对“bag 未录完/数据文件为空/话题类型不符”都给出明确的中文可读报错。
#   入口：命令行 `hope_bag_to_csv --bag <dir> --output <csv> [--topic /poses] [--ball-index 0]`。
# =============================================================================

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

from geometry_msgs.msg import PointStamped, PoseArray
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


def _detect_storage_id(bag_dir: Path) -> str:
    # 识别 rosbag2 存储后端：优先从 metadata.yaml 读 storage_identifier；
    # 读不到再按数据文件后缀猜（*.mcap→mcap，*.db3→sqlite3）。
    metadata = bag_dir / "metadata.yaml"
    if metadata.is_file():
        text = metadata.read_text(errors="replace")
        match = re.search(r"storage_identifier:\s*([A-Za-z0-9_]+)", text)
        if match:
            return match.group(1)

    mcap_files = list(bag_dir.glob("*.mcap"))
    db3_files = list(bag_dir.glob("*.db3"))
    if mcap_files:
        return "mcap"
    if db3_files:
        return "sqlite3"
    raise RuntimeError(
        f"Could not detect rosbag storage type in {bag_dir}. "
        "Expected metadata.yaml, *.mcap, or *.db3."
    )


def _check_bag_files_look_real(bag_dir: Path) -> None:
    # 录制完整性预检：既要有数据文件，又不能是 0 字节（录制中途 kill 常见坑）。
    data_files = list(bag_dir.glob("*.mcap")) + list(bag_dir.glob("*.db3"))
    if not data_files:
        raise RuntimeError(
            f"No bag data file found in {bag_dir}. "
            "The recording may not have finished correctly."
        )

    zero_files = [p.name for p in data_files if p.stat().st_size == 0]
    if zero_files:
        raise RuntimeError(
            f"Bag data file is empty in {bag_dir}: {', '.join(zero_files)}. "
            "This usually means recording did not finish correctly. Re-record this toss."
        )


def bag_point_topic_to_csv(bag_path: str, topic: str, output_csv: str, ball_index: int = 0) -> int:
    # 主转换函数：打开 bag → 顺序读取指定话题 → 抽取球位与时间 → 按时间排序 → 写 CSV。
    # 返回写入的行数。
    bag_dir = Path(bag_path)
    if not bag_dir.exists():
        raise RuntimeError(f"Bag path does not exist: {bag_path}")
    if not bag_dir.is_dir():
        raise RuntimeError(f"Bag path is not a directory: {bag_path}")

    _check_bag_files_look_real(bag_dir)
    storage_id = _detect_storage_id(bag_dir)

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id=storage_id),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    # 校验目标话题存在，并按其消息类型动态拿到反序列化类型。
    topic_types = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
    if topic not in topic_types:
        raise RuntimeError(f"Topic {topic!r} not found in bag {bag_path!r}")

    msg_type = get_message(topic_types[topic])
    rows = []

    # 顺序遍历 bag 里所有消息，只保留目标话题的帧。
    while reader.has_next():
        current_topic, data, bag_time_ns = reader.read_next()
        if current_topic != topic:
            continue
        msg = deserialize_message(data, msg_type)
        if isinstance(msg, PoseArray):
            # The standard bringup stream: ball at poses[ball_index] (pose_to_posearray contract).
            if len(msg.poses) <= ball_index:
                continue  # frame without the ball slot (e.g. adapter still warming up)
            point = msg.poses[ball_index].position
        elif isinstance(msg, PointStamped):
            point = msg.point
        else:
            raise RuntimeError(
                f"Topic {topic!r} is {topic_types[topic]!r}; expected geometry_msgs/PoseArray "
                "(the standard /poses stream) or geometry_msgs/PointStamped."
            )

        # 时间戳：消息头非零则用消息头（更接近采集时刻）；否则退回 bag 接收时间（纳秒→秒）。
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            t = bag_time_ns * 1e-9

        rows.append((t, point.x, point.y, point.z))

    if not rows:
        raise RuntimeError(f"No messages found on topic {topic!r} in bag {bag_path!r}")

    # 按时间升序排序后写出（bag 内顺序通常已按接收时间，但消息头时间可能乱序，故显式排序）。
    rows.sort(key=lambda r: r[0])
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "x", "y", "z"])
        writer.writerows(rows)
    return len(rows)


def main(argv=None) -> None:
    # 命令行入口（对应 setup.py 里的 hope_bag_to_csv 脚本）：解析参数并调用转换函数。
    parser = argparse.ArgumentParser(
        description="Convert a rosbag2 ball stream (/poses PoseArray or a PointStamped topic) "
        "into HOPE calibration CSV (t,x,y,z)."
    )
    parser.add_argument("--bag", required=True, help="Path to rosbag2 directory.")
    parser.add_argument(
        "--topic",
        default="/poses",
        help="Topic to export: geometry_msgs/PoseArray (the standard bringup /poses stream) or a "
        "geometry_msgs/PointStamped topic. Default: /poses",
    )
    parser.add_argument(
        "--ball-index",
        type=int,
        default=0,
        help="PoseArray slot carrying the ball (matches the planner's ball_pose_index). Default: 0",
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args(argv)

    count = bag_point_topic_to_csv(args.bag, args.topic, args.output, ball_index=args.ball_index)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
