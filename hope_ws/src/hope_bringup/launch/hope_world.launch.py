"""Publish the HOPE world-frame static transforms from config."""

# =============================================================================
# 【中文说明】按配置发布 HOPE 世界坐标系下的一组静态 TF（球桌/球网/半场/地面/击球平面等）
# -----------------------------------------------------------------------------
#   从 config/hope_world_frame.yaml 读取各地标坐标，为每个地标起一个 tf2_ros 的
#   static_transform_publisher，把它们作为 world 的子坐标系发布，供 RViz 可视化与
#   下游对齐使用。其中 robot_mocap→robot_base_link 用配置里的实测外参(xyz/rpy)。
# =============================================================================

from pathlib import Path

import yaml
from launch import LaunchDescription
from launch_ros.actions import Node


def _load_world_config():
    # 读取本包 config/hope_world_frame.yaml 的 hope_world 段。
    config_path = Path(__file__).resolve().parent.parent / "config" / "hope_world_frame.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["hope_world"]


def _static_tf(parent_frame, child_frame, xyz, rpy):
    # 生成一个静态 TF 发布节点：parent→child，平移 xyz(米)、旋转 rpy(弧度)。
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=f"{child_frame}_static_tf".replace("/", "_"),
        arguments=[
            "--x", str(xyz[0]),
            "--y", str(xyz[1]),
            "--z", str(xyz[2]),
            "--roll", str(rpy[0]),
            "--pitch", str(rpy[1]),
            "--yaw", str(rpy[2]),
            "--frame-id", parent_frame,
            "--child-frame-id", child_frame,
        ],
    )


def generate_launch_description():
    # 解析配置：坐标系名、各地标坐标、动捕→base_link 外参、击球平面 x。
    config = _load_world_config()
    frames = config["frames"]
    landmarks = config["landmarks_m"]
    offset = config["mocap_to_base_link"]
    x_hit = config["planner"]["x_hit"]

    # 逐地标建立 world→地标 的静态 TF；最后一条是 robot_mocap→base_link 的实测外参。
    world = frames["world"]
    nodes = [
        _static_tf(world, frames["table_center"], landmarks["table_center"], [0.0, 0.0, 0.0]),
        _static_tf(world, frames["robot_half_center"], landmarks["robot_half_center"], [0.0, 0.0, 0.0]),
        _static_tf(world, frames["opponent_half_center"], landmarks["opponent_half_center"], [0.0, 0.0, 0.0]),
        _static_tf(world, frames["net_center"], landmarks["net_center"], [0.0, 0.0, 0.0]),
        _static_tf(world, frames["floor_origin"], landmarks["floor_origin"], [0.0, 0.0, 0.0]),
        _static_tf(world, frames["virtual_hit_plane"], [x_hit, 0.0, 0.0], [0.0, 0.0, 0.0]),
        _static_tf(frames["robot_mocap"], frames["robot_base_link"], offset["xyz"], offset["rpy"]),
    ]
    return LaunchDescription(nodes)
