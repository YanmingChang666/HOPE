# =============================================================================
# 【中文说明】hope_planner 的 ament_python 打包脚本（colcon build 用）
# -----------------------------------------------------------------------------
#   声明包名、要打包的 Python 包、随包安装的资源(config/launch)，以及可执行入口。
#   构建后 `ros2 run hope_planner hope_planner_node` 即启动规划节点。
# =============================================================================

import os
from glob import glob

from setuptools import find_packages, setup

package_name = "hope_planner"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),   # 打包 Python 包，排除 test 目录
    data_files=[
        # 向 ament 资源索引注册本包（让 ros2 能发现它）
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # 把 config/*.yaml 与 launch/*.launch.py 一并装到 share 下，供运行时/launch 读取
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HOPE Maintainers",
    maintainer_email="maintainer@example.com",
    description="HOPE no-spin model-based racket planner.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        # 可执行命令入口：名称 = 模块:函数
        "console_scripts": [
            "hope_planner_node = hope_planner.node:main",   # 规划主节点
            "hope_bag_to_csv = hope_planner.bag_to_csv:main",  # bag→CSV 标定工具
        ],
    },
)
