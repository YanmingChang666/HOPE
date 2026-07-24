"""Launch and visualize the HOPE table-tennis match scene (physics + visualization, no policy).

Builds the full court (floor, table, net, ball, Agibot A3) in the HOPE frame, serves a ball each reset,
and steps the simulation holding the robot's default standing pose (zero action). Use this to verify the
physics (ball flight, drag, table/net bounce) and the scene layout before training a policy.

Run inside your Isaac Lab GPU environment after ``source setup_train_env.sh`` (which defines
``isaac_py``, the Isaac Python launcher with the working-tree PYTHONPATH):

    # interactive window (default: 1 env, Agibot A3, robot free-standing, aerodynamics on)
    isaac_py scripts/play_table_tennis.py

    # the Unitree G1 scene instead of the A3
    isaac_py scripts/play_table_tennis.py --robot g1

    # several courts at once
    isaac_py scripts/play_table_tennis.py --num_envs 9

    # pin the robot upright (stable view of the ball physics while no balance policy exists)
    isaac_py scripts/play_table_tennis.py --fix_base

    # compare flight with/without aerodynamic drag
    isaac_py scripts/play_table_tennis.py --disable_aero

This uses the standard Isaac Lab ``AppLauncher`` standalone pattern (no Hydra), so it runs without
a trained checkpoint. The shipped ball model is no-spin, so there is no Magnus option.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize the HOPE table-tennis scene.")
parser.add_argument("--robot", choices=["a3", "g1"], default="a3",
                    help="Which robot's table-tennis scene to launch (a3 = Agibot A3, g1 = Unitree G1).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel courts to spawn.")
parser.add_argument("--fix_base", action="store_true", help="Pin the robot pelvis (stable visualization).")
parser.add_argument("--disable_aero", action="store_true", help="Disable ball aerodynamic drag.")
parser.add_argument("--steps", type=int, default=0, help="Stop after N control steps (0 = run until window closed).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse / Isaac Sim first; all isaaclab.* / task imports must come after this.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import torch

    import whole_body_tracking.tasks  # noqa: F401 -- registers the Gym tasks (import_packages)

    if args_cli.robot == "g1":
        from whole_body_tracking.tasks.table_tennis.config.unitree_g1.table_tennis_env_cfg import (
            G1TableTennisEnvCfg as RobotTableTennisEnvCfg,
        )

        task_id = "HOPE-TableTennis-UnitreeG1-v0"
    else:
        from whole_body_tracking.tasks.table_tennis.config.agibot_a3.table_tennis_env_cfg import (
            AgibotA3TableTennisEnvCfg as RobotTableTennisEnvCfg,
        )

        task_id = "HOPE-TableTennis-AgibotA3-v0"

    env_cfg = RobotTableTennisEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    if args_cli.fix_base:
        # fix_base is a URDF-spawn (A3) option; the G1 spawns from USD, where it does not apply.
        if hasattr(env_cfg.scene.robot.spawn, "fix_base"):
            env_cfg.scene.robot.spawn.fix_base = True
        else:
            print("[play_table_tennis] --fix_base is not supported for this spawn type; ignoring.")
    if args_cli.disable_aero:
        env_cfg.ball_aerodynamics.enabled = False

    env = gym.make(task_id, cfg=env_cfg)
    print(f"[play_table_tennis] launched '{task_id}' with {env.unwrapped.num_envs} env(s).")
    print(f"[play_table_tennis] ball aerodynamics active: {getattr(env.unwrapped, '_aero_active', False)}")

    obs, _ = env.reset()
    # Zero action in the joint-position-with-default-offset space = hold the standing pose.
    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            obs, rew, terminated, truncated, info = env.step(zero_action)
        step += 1
        if args_cli.steps and step >= args_cli.steps:
            break

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
