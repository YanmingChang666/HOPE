"""Launch the HOPE planner node with the default config."""

# =============================================================================
# 【中文说明】只启动规划节点本身的 launch（用默认 config/hope_planner.yaml 覆盖参数）
# -----------------------------------------------------------------------------
#   适用于“球流已由别处提供(/poses 已存在)”的场景，只需拉起 hope_planner 节点。
#   若要连动捕客户端+适配器一起启动，用 hope_bringup 包的 hope_bringup.launch.py。
# =============================================================================

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # 从已安装的 share 目录定位默认参数文件，再作为 parameters 传给节点。
    config = Path(get_package_share_directory("hope_planner")) / "config" / "hope_planner.yaml"
    return LaunchDescription([
        Node(
            package="hope_planner",
            executable="hope_planner_node",
            name="hope_planner",
            output="screen",
            parameters=[str(config)],
        ),
    ])
