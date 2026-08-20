"""Generic HOPE bringup: motion capture -> planner.

Starts the racket planner and its ball source. By default it launches the
``vrpn_mocap`` VRPN client (configurable server address, default ``localhost``)
plus the ``pose_to_posearray`` adapter so the planner runs against a real
motion-capture stream. For testing without mocap, set ``use_fake_ball:=true``
to publish a synthetic ``/poses`` stream instead.

The planner subscribes to ``poses_topic`` (a ``geometry_msgs/PoseArray`` with
the ball at ``ball_pose_index``, default 0). The VRPN client publishes one
``PoseStamped`` topic per tracker, so the ``pose_to_posearray`` node aggregates
the configured tracker topic(s) into that PoseArray — set ``ball_pose_topic`` to
your ball tracker's pose topic (check with ``ros2 topic list | grep vrpn``).
The bundled ``client.launch.yaml`` forces ``multi_sensor: true``, and in that
mode the driver names every pose topic ``pose_id_<N>`` — hence the default
``/vrpn_mocap/ball/pose_id_0`` for a tracker named ``ball``.
``fake_ball_publisher`` publishes the PoseArray form directly.

Examples::

    # Real mocap on this machine (ball tracker named "ball"):
    ros2 launch hope_bringup hope_bringup.launch.py mocap_server:=localhost

    # Real mocap on another host, different tracker topic:
    ros2 launch hope_bringup hope_bringup.launch.py mocap_server:=mocap.local \\
        mocap_port:=3883 ball_pose_topic:=/vrpn_mocap/Ball/pose_id_0

    # No mocap, synthetic ball for a smoke test:
    ros2 launch hope_bringup hope_bringup.launch.py use_fake_ball:=true
"""

# =============================================================================
# 【中文说明】通用总启动：动捕/球源 → 规划器（把整条链路一键拉起）
# -----------------------------------------------------------------------------
#   两种运行模式（由 use_fake_ball 切换）：
#     · 真实动捕(默认 use_fake_ball:=false)：启动 vrpn_mocap 客户端 + pose_to_posearray
#       适配器，把每个 tracker 的 PoseStamped 聚合成规划器要的 /poses(PoseArray)。
#     · 仿真球(use_fake_ball:=true)：不启动动捕，改用 fake_ball_publisher 直接发合成 /poses，
#       用于无动捕环境下的冒烟测试。
#   关键参数：mocap_server/mocap_port(VRPN 服务器)、ball_pose_topic(球 tracker 的位姿话题)。
#   易踩坑：随包的 client.launch.yaml 强制 multi_sensor:=true，此模式下话题名带 pose_id_<N>
#           后缀 → 名为 "ball" 的 tracker 实际发布 /vrpn_mocap/ball/pose_id_0（故为默认值）。
#           用 `ros2 topic list | grep vrpn` 确认你现场真实的话题名。
# =============================================================================

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # 读取可在命令行覆盖的 launch 参数（下方 DeclareLaunchArgument 定义了默认值与说明）。
    mocap_server = LaunchConfiguration("mocap_server")
    mocap_port = LaunchConfiguration("mocap_port")
    use_fake_ball = LaunchConfiguration("use_fake_ball")
    ball_pose_topic = LaunchConfiguration("ball_pose_topic")

    planner_config = Path(get_package_share_directory("hope_planner")) / "config" / "hope_planner.yaml"

    # 组件一：VRPN 动捕客户端（仅在非仿真模式启动），把服务器地址/端口透传进去。
    vrpn_client = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("vrpn_mocap"), "launch", "client.launch.yaml"])
        ),
        launch_arguments={"server": mocap_server, "port": mocap_port}.items(),
        condition=UnlessCondition(use_fake_ball),
    )

    # 组件二：真实动捕适配器（仅非仿真模式）。把逐 tracker 的 PoseStamped 聚合成规划器要的
    # /poses PoseArray（球在 index 0；触发消息的头时间戳原样透传——用随包 VRPN 驱动时那是
    # ROS 端接收时间，除非开了 use_vrpn_timestamps）。
    # 注意嵌套列表 [[ball_pose_topic]]：launch_ros 会把“扁平”的 substitution 列表拼接成单个
    # 字符串，这会违反节点参数的 STRING_ARRAY 类型；写成“列表套列表”才求值为字符串数组。
    pose_adapter = Node(
        package="hope_bringup",
        executable="pose_to_posearray",
        name="pose_to_posearray",
        output="screen",
        parameters=[{"input_topics": [[ball_pose_topic]], "trigger_index": 0}],
        condition=UnlessCondition(use_fake_ball),
    )

    # 组件三：合成球发布器（仅仿真模式），直接发 /poses，替代动捕做冒烟测试。
    fake_ball = Node(
        package="hope_bringup",
        executable="fake_ball_publisher",
        name="fake_ball_publisher",
        output="screen",
        condition=IfCondition(use_fake_ball),
    )

    # 组件四：规划节点本体（两种模式都启动），加载默认参数文件。
    planner = Node(
        package="hope_planner",
        executable="hope_planner_node",
        name="hope_planner",
        output="screen",
        parameters=[str(planner_config)],
    )

    # 汇总：先声明四个 launch 参数（默认值+说明），再列出上面四个组件节点。
    return LaunchDescription([
        DeclareLaunchArgument(
            "mocap_server", default_value="localhost",
            description="VRPN motion-capture server IP/hostname."),
        DeclareLaunchArgument(
            "mocap_port", default_value="3883",
            description="VRPN motion-capture server port."),
        DeclareLaunchArgument(
            "use_fake_ball", default_value="false",
            description="Publish a synthetic /poses ball stream instead of starting vrpn_mocap."),
        DeclareLaunchArgument(
            "ball_pose_topic", default_value="/vrpn_mocap/ball/pose_id_0",
            description="The ball tracker's PoseStamped topic aggregated into /poses. The bundled "
                        "VRPN client runs with multi_sensor:=true, which names topics "
                        "pose_id_<N>; a tracker named 'ball' therefore publishes "
                        "/vrpn_mocap/ball/pose_id_0."),
        vrpn_client,
        pose_adapter,
        fake_ball,
        planner,
    ])
