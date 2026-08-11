"""Hydra training entry for the HOPE Agibot A3 policy.

Single task, single algo. Build the ``HOPE-PingPong-AgibotA3-v0`` environment (111-D actor
observation, privileged critic, 50 Hz control, ``wrap_teleport: false``), a rsl_rl PPO runner, and
train. Checkpoints are written locally (periodic every ``save_interval`` and a final one). There is
no Weights & Biases, no external logging service, no gate / lineage / curriculum machinery.

Usage:
    python scripts/train.py task=HOPEPingPong algo=ppo headless=true

Override any field on the CLI, e.g.:
    python scripts/train.py task=HOPEPingPong num_envs=2048 max_iterations=20000 seed=1 \
        motion_file=/abs/hope_forehand.npz motion_file_2=/abs/hope_backhand.npz

Tune training by editing cfg/task/HOPEPingPong.yaml and cfg/algo/ppo.yaml.
"""

import os
import pathlib
import sys

import hydra
from omegaconf import OmegaConf


def _repo_root() -> pathlib.Path:
    """Repo root = the directory that contains ``hope_training/`` (walk up from this file).

    中文说明：定位「仓库根目录」。从本文件出发向上逐级查找，找到第一个包含
    ``hope_training/`` 子目录的父目录即为根目录。这样无论从哪个工作目录启动训练，
    动作片段（.npz）等「相对仓库根」的路径都能被稳定解析。找不到时退回到上溯两级的目录。
    """
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hope_training").is_dir():
            return parent
    return here.parents[2]


def _resolve_motion_path(value: str) -> str:
    """Resolve a clip path: absolute / cwd-relative first, then repo-root-relative.

    中文说明：把一个动作片段路径解析成真实存在的绝对路径。优先按「绝对路径 / 相对当前
    工作目录」查找；找不到再按「相对仓库根目录」查找。两者都不存在时，返回「仓库根 +
    原路径」这个候选，以便后续报错信息指向一个稳定位置，方便用户排查。
    """
    p = pathlib.Path(str(value))
    if p.is_file():
        return str(p.resolve())
    rooted = _repo_root() / value
    if rooted.is_file():
        return str(rooted.resolve())
    # Return the repo-root candidate so the error message points at a stable location.
    return str(rooted)


def _resolve_motion_sources(cfg) -> list[str]:
    """Return the list of local clip paths [forehand, backhand] (CLI overrides the task cfg).

    中文说明：收集本次训练要模仿的参考动作片段列表 [正手, 反手]。取值优先级为「命令行
    传入的 motion_file / motion_file_2」高于「任务 YAML 里的默认值」。片段 0 = 正手
    (forehand)，片段 1 = 反手 (backhand)；反手可省略（只训练正手时）。函数会逐个做路径
    解析与「文件是否存在」的校验，任何一个缺失都会立即抛错，避免训练跑到中途才发现缺片段。
    """
    primary = cfg.motion_file if cfg.motion_file is not None else cfg.task.get("motion_file")
    secondary = cfg.motion_file_2 if cfg.motion_file_2 is not None else cfg.task.get("motion_file_2")
    clips = [primary]
    if secondary is not None:
        clips.append(secondary)
    resolved = [_resolve_motion_path(c) for c in clips if c is not None]
    if not resolved:
        raise RuntimeError(
            "No motion clip configured. Set motion_file (and motion_file_2) on the CLI or in "
            "cfg/task/HOPEPingPong.yaml."
        )
    for clip in resolved:
        if not pathlib.Path(clip).is_file():
            raise FileNotFoundError(
                f"motion clip not found: {clip}\nProvide your own clips or the placeholder clips "
                "under hope_training/motions/preprocessed/ (see docs/REPLACE_MOTIONS.md)."
            )
    return resolved


def _set_dotted(obj, dotted: str, value, applied: list, where: str) -> None:
    """Set ``obj.<a>.<b>... = value`` if the attribute chain exists; else warn and skip.

    中文说明：按「点分路径」安全地给配置对象赋值，例如把字符串
    ``"commands.motion.wrap_teleport"`` 逐级解析成 ``obj.commands.motion.wrap_teleport``
    并写入 value。中途任何一层属性不存在，就打印警告并跳过（不抛异常），从而允许在不同
    机器人 / 不同 env 配置间复用同一套覆盖逻辑。成功写入的项会追加到 ``applied`` 列表，
    最后统一打印，便于核对本次到底改了哪些字段。
    """
    parts = dotted.split(".")
    node = obj
    for attr in parts[:-1]:
        if not hasattr(node, attr):
            print(f"[train.py] WARNING: {where}: '{dotted}' — no attribute '{attr}'; skipped.", flush=True)
            return
        node = getattr(node, attr)
    leaf = parts[-1]
    if not hasattr(node, leaf):
        print(f"[train.py] WARNING: {where}: '{dotted}' — no attribute '{leaf}'; skipped.", flush=True)
        return
    setattr(node, leaf, value)
    applied.append(f"{dotted} = {value}")


def _apply_domain_rand(env_cfg, dr, applied: list) -> None:
    """Apply the shared link-mass / PD-gain randomization knobs.

    The event terms are named ``events.link_mass`` and ``events.pd_gains`` in
    :class:`HOPEPingPongEnvCfg.EventCfg` — the override MUST target those exact fields
    (see ``tests/test_domain_rand_overrides.py``). Semantics per range knob:
      * absent          -> keep the env-cfg default;
      * ``null``        -> disable the event entirely (set the term to None);
      * ``[lo, hi]``    -> override the distribution parameters.

    中文说明：应用「域随机化 (domain randomization)」开关，用于提升 sim-to-real 鲁棒性。
    目前支持两个共享旋钮：``link_mass``（连杆质量缩放）与 ``pd_gains``（PD 刚度/阻尼缩放）。
    每个旋钮的取值语义：
      * 不填          -> 保持 env 配置里的默认随机范围；
      * ``null``      -> 整个关闭该随机化事件（把对应 EventTerm 置为 None）；
      * ``[lo, hi]``  -> 用新的上下界覆盖随机分布参数。
    注意：这里的字段名 ``events.link_mass`` / ``events.pd_gains`` 必须与 env 配置里
    EventCfg 的字段一一对应，否则会跳过并告警（有单元测试守护这一契约）。
    """
    if dr is None:
        return
    events = getattr(env_cfg, "events", None)
    if events is None:
        return

    def _apply(range_key: str, event_name: str, param_keys: tuple[str, ...]) -> None:
        if range_key not in dr:
            return
        if not hasattr(events, event_name):
            print(
                f"[train.py] WARNING: domain_rand.{range_key}: events.{event_name} does not "
                "exist on this env cfg; skipped.",
                flush=True,
            )
            return
        rng = dr.get(range_key)
        if rng is None:
            if getattr(events, event_name) is not None:
                setattr(events, event_name, None)
                applied.append(f"events.{event_name} = None (disabled)")
            return
        term = getattr(events, event_name)
        if term is None:
            print(
                f"[train.py] WARNING: domain_rand.{range_key}: events.{event_name} is already "
                "disabled in the env cfg; range ignored.",
                flush=True,
            )
            return
        lo, hi = float(rng[0]), float(rng[1])
        for key in param_keys:
            term.params[key] = (lo, hi)
        applied.append(f"events.{event_name} = {(lo, hi)}")

    _apply("link_mass_range", "link_mass", ("mass_distribution_params",))
    _apply(
        "pd_gain_range",
        "pd_gains",
        ("stiffness_distribution_params", "damping_distribution_params"),
    )


def _apply_task_overrides(env_cfg, cfg, applied: list) -> None:
    """Apply the launcher-level knobs + generic dotted-path overrides from the task cfg.

    中文说明：把「任务 YAML (cfg/task/*.yaml)」里的启动级旋钮写回到已解析的 env 配置上，
    使得改超参无需修改 Python。共处理四类覆盖：
      1. ``env.episode_length_s``          —— 单个 episode 时长（多拍/多回合）；
      2. ``motion.wrap_teleport``          —— 片段循环时是否把机器人瞬移回起点
                                             （连续多回合必须为 False，让策略学会击球之间的恢复）；
      3. ``domain_rand``                   —— 交给 :func:`_apply_domain_rand` 处理；
      4. 通用 ``overrides`` 映射            —— 任意「点分路径 -> 值」，交给 :func:`_set_dotted`。
    """
    task = cfg.task
    # episode length (top-level on ManagerBasedRLEnvCfg).
    env_block = task.get("env")
    if env_block is not None and env_block.get("episode_length_s") is not None:
        _set_dotted(env_cfg, "episode_length_s", float(env_block.get("episode_length_s")), applied, "env")
    # continuous multi-rally lifecycle: no teleport on clip wrap.
    motion_block = task.get("motion")
    if motion_block is not None and motion_block.get("wrap_teleport") is not None:
        _set_dotted(
            env_cfg, "commands.motion.wrap_teleport", bool(motion_block.get("wrap_teleport")), applied, "motion"
        )
    # domain randomization.
    _apply_domain_rand(env_cfg, task.get("domain_rand"), applied)
    # generic overrides map (dotted attribute paths -> value).
    overrides = task.get("overrides")
    if overrides:
        for dotted, value in OmegaConf.to_container(overrides, resolve=True).items():
            _set_dotted(env_cfg, str(dotted), value, applied, "overrides")


def _run(cfg):
    # ============================================================================================
    # 中文说明：整个训练的「主流程」。此函数在 Isaac Sim 已经启动之后被调用，按顺序完成：
    #   1) 根据已注册的 Gym 任务解析 env 配置，并叠加任务级覆盖；
    #   2) 载入参考动作片段（正手 / 反手 .npz）作为模仿目标；
    #   3) 从 cfg/algo/ppo.yaml 构造 rsl_rl 的 PPO runner 配置；
    #   4) 计算本地日志 / checkpoint 目录；
    #   5) 创建仿真环境、（可选）录像、并用 rsl_rl 向量化包装器包裹；
    #      —— 期间有两道硬性「关节顺序 / 观测契约」校验闸门，防止训练出无法部署的策略；
    #   6) （可选）从已有 checkpoint 断点续训；
    #   7) 落盘解析后的配置并调用 runner.learn(...) 正式开始 PPO 训练。
    # 训练产物：logs/rsl_rl/<experiment>/<时间戳>/ 下的周期性与最终 checkpoint（无 W&B/TensorBoard）。
    # ============================================================================================
    import gymnasium as gym
    import torch
    from datetime import datetime

    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import whole_body_tracking.tasks  # noqa: F401  -- registers the gym task
    from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    # 1) environment cfg from the registered gym task + task-cfg overrides.
    #    中文：由 Gym 任务 id（如 HOPE-PingPong-UnitreeG1-v0）解析出完整 env 配置，
    #    再把 seed / device / 任务级覆盖写入。num_envs 决定并行环境数（默认数千个）。
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    applied: list = []
    _apply_task_overrides(env_cfg, cfg, applied)
    env_cfg.seed = int(cfg.seed)
    env_cfg.sim.device = str(cfg.device)
    print(f"[train.py] task={task_id} num_envs={num_envs} — applied {len(applied)} task override(s):", flush=True)
    for line in applied:
        print(f"[train.py]     {line}", flush=True)

    # 2) reference motion clips (local .npz; clip 0 = forehand, clip 1 = backhand).
    #    中文：确定参考动作片段并写入 commands.motion.motion_file。这些 .npz 存的是参考轨迹
    #    （关节角/关节速 + 被跟踪连杆的世界位姿/速度），是「动作模仿(imitation)」奖励的目标。
    #    单片段则直接传路径，多片段则传列表，由 MotionLoader 拼到同一条时间轴上。
    motion_files = _resolve_motion_sources(cfg)
    for i, mf in enumerate(motion_files):
        print(f"[train.py] motion clip {i}: {mf}", flush=True)
    env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]

    # 3) PPO runner cfg from cfg/algo/ppo.yaml.
    #    中文：把 cfg/algo/ppo.yaml 里的 PPO 超参（网络维度、学习率、KL 自适应、GAE、
    #    clip、熵系数、rollout 长度、最大迭代数等）映射成 rsl_rl 的 RslRlOnPolicyRunnerCfg。
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(algo, str(cfg.task.experiment_name)))
    agent_cfg.seed = int(cfg.seed)
    agent_cfg.device = str(cfg.device)
    if cfg.max_iterations is not None:
        agent_cfg.max_iterations = int(cfg.max_iterations)
    if cfg.run_name is not None:
        agent_cfg.run_name = str(cfg.run_name)

    # 4) local logging directory.
    #    中文：按 logs/rsl_rl/<experiment>/<时间戳>[_<run_name>] 生成本地日志与 checkpoint 目录。
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, log_dir)
    print(f"[train.py] experiment={agent_cfg.experiment_name} | log_dir={log_dir}", flush=True)

    # 5) build env, (optionally) record video, wrap for rsl_rl.
    #    中文：真正实例化仿真环境（此时才会 spawn 机器人 USD、建关节表）。若开了 video 就用
    #    rgb_array 渲染并按间隔录像；随后经过两道校验闸门，最后用 RslRlVecEnvWrapper 包装。
    render_mode = "rgb_array" if cfg.video else None
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)

    # HARD GATE (train time): the articulation's joint enumeration fixes the obs/action column
    # order of everything this run learns. It must equal the canonical deploy joint order — the
    # same check export_onnx.py applies — so a permuted asset fails BEFORE training, not after a
    # full run when the stale checkpoint meets the export gate.
    #    中文【硬闸门①：关节顺序】：articulation 枚举出的关节顺序决定了观测/动作向量每一列的含义。
    #    它必须与「部署时的规范关节顺序」完全一致，否则训练出来的策略在真机上列会错位、无法部署。
    #    这里在训练开始前就做校验（与 export_onnx.py 同一套检查），一旦顺序被打乱立即报错，
    #    避免白跑一整轮后在导出阶段才失败。
    from whole_body_tracking.utils.action_adapter_config import load_joint_order_for_dof

    _joint_names = list(env.unwrapped.scene["robot"].data.joint_names)
    _expected_tuple, _canon_file = load_joint_order_for_dof(len(_joint_names))
    _expected_order = list(_expected_tuple)
    if _joint_names != _expected_order:
        raise RuntimeError(
            f"Articulation joint order does not match the canonical deploy joint order ({_canon_file}).\n"
            f"  articulation: {_joint_names}\n"
            f"  canonical:    {_expected_order}\n"
            "Fix the URDF/USD so its joint enumeration matches the canonical order before "
            f"training, or paste the articulation order above into {_canon_file} (single source of "
            "truth). A policy trained on a permuted enumeration cannot be deployed."
        )
    print("[train.py] joint-order gate: articulation matches the canonical deploy order.", flush=True)

    # Validate the 111-D actor observation contract when the task declares one (guarded import).
    #    中文【硬闸门②：观测契约】：校验 actor 观测的维度与字段布局符合任务声明的契约
    #    （A3 为 111 维 hope_pingpong；G1 为 105 维 hope_pingpong_g1）。保证策略输入格式
    #    与部署端一致。校验器缺失时安全跳过。
    expected_contract = cfg.task.get("actor_obs_contract")
    if expected_contract is not None:
        try:
            from whole_body_tracking.tasks.tracking.actor_observation_contract import (
                validate_actor_observation_contract,
            )

            contract = validate_actor_observation_contract(env.unwrapped, str(expected_contract))
            print(
                f"[train.py] actor observation contract validated: {contract.name} "
                f"({contract.total_dim}D)",
                flush=True,
            )
        except ImportError:
            print("[train.py] NOTE: actor_observation_contract validator not available; skipping.", flush=True)

    if cfg.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % int(cfg.video_interval) == 0,
            video_length=int(cfg.video_length),
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    #    中文：构造 HOPE 版 PPO runner（继承 rsl_rl 的 OnPolicyRunner，仅把日志写入替换为
    #    离线空实现——不依赖 W&B / TensorBoard），并把当前 git 仓库信息记进日志便于复现。
    runner = HOPEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # 6) optional resume from a local checkpoint (strict: weights + optimizer + iteration counter).
    #    中文：可选断点续训。给了 checkpoint_path 就严格载入「网络权重 + 优化器状态 + 迭代计数」，
    #    从上次的迭代数继续；不给则从随机初始化开始。
    ckpt = getattr(cfg, "checkpoint_path", None)
    if ckpt is not None:
        ckpt = os.path.abspath(str(ckpt))
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"[train.py] checkpoint_path does not exist: {ckpt}")
        runner.load(ckpt)
        print(f"[train.py] resumed from checkpoint: {ckpt}", flush=True)

    # 7) dump the resolved configuration + train.
    #    中文：把最终解析出的 env / agent 配置落盘（便于复现），然后调用 runner.learn(...)
    #    正式开始 PPO 训练循环：采样 rollout → 计算 GAE 优势 → 多轮 minibatch 更新 actor/critic
    #    → 周期性保存 checkpoint，直到达到 max_iterations。init_at_random_ep_len=True 让各并行
    #    环境的初始 episode 进度随机错开，避免所有环境同时重置。
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    # 中文说明：Hydra 命令行入口。cfg 由 cfg/train.yaml + task/algo 子配置 + 命令行覆盖合并而来。
    # 关键点：必须「先启动 Isaac Sim（AppLauncher），再 import 任何 isaaclab 模块」，否则会报错；
    # 因此把训练主体放在 _run() 里、在 App 启动之后才调用。这里还清空了 sys.argv，防止 Isaac Kit
    # 去解析 Hydra 的 task=.../algo=... 参数而报错。无论成功失败都会在 finally 里关闭仿真 App。
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Launch Isaac Sim BEFORE importing isaaclab modules. Clear argv so Kit does not try to parse
    # Hydra's task=.../algo=... overrides.
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=bool(cfg.video))
    simulation_app = app_launcher.app

    failed = False
    try:
        _run(cfg)
    except Exception:
        import traceback

        print("\n[train.py] ERROR during run:", flush=True)
        traceback.print_exc()
        failed = True
    finally:
        simulation_app.close()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
