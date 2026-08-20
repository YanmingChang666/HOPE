"""Unitree G1 — the single HOPE whole-body task (parallel to the Agibot A3 twin).

Same task wiring as ``config/agibot_a3/hope_env_cfg.py`` (motion imitation over a forehand +
backhand clip pair, the ping-pong racket-target goal, the actor observation + privileged critic,
the eleven reward terms, the clamped joint-position residual action, and light domain
randomization), specialized for the G1:

* 29 controllable DOF (no head) → 105-D actor observation / 29-D action (``hope_pingpong_g1``);
* ``passive_joint_names=()`` (there is no passive neck to hold);
* G1 body names (lowercase ``_link``; root ``pelvis``; anchor ``torso_link``);
* the racket-mount FK fields are set EXPLICITLY (the A3 twin silently relied on the A3 dataclass
  defaults — the G1 must override them for the right-wrist paddle);
* default_q / action_scale / joint clamp come from the G1 shared deploy ``action_adapter.yaml``.

Control runs at 50 Hz. Motion clips default to the G1 placeholders under
``hope_training/motions/preprocessed`` — replace them with your own retargeted G1 clips.
"""

from __future__ import annotations

import copy
import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.g1 import (
    G1_ANCHOR_BODY,
    G1_CFG,
    G1_FEET_BODIES,
    G1_MOUNT_NORMAL_AXIS,
    G1_MOUNT_OFFSET,
    G1_MOUNT_QUAT,
    G1_PASSIVE_HEAD_JOINT_NAMES,
    G1_RACKET_BODY,
    G1_TRACKED_BODIES,
    G1_UPPER_TRACKED,
    G1_WRIST_BODY,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import MySceneCfg
from whole_body_tracking.utils.action_adapter_config import load_g1_action_adapter_config


def _find_motion_clip(name: str) -> str:
    """Locate a placeholder clip under ``hope_training/motions/preprocessed`` (walk up from here)."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(14):
        cand = os.path.join(d, "hope_training", "motions", "preprocessed", name)
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.join("hope_training", "motions", "preprocessed", name)


FOREHAND_CLIP = _find_motion_clip("hope_g1_forehand.npz")
BACKHAND_CLIP = _find_motion_clip("hope_g1_backhand.npz")


def _apply_offline_ground(scene) -> None:
    """Swap the ``plane`` terrain — which downloads a ground-plane USD from the Isaac asset server —
    for a procedurally-generated flat terrain that needs NO network/assets.

    Opt-in via the ``HOPE_OFFLINE_GROUND`` env var, for machines that cannot reach the remote Isaac
    asset root (``get_assets_root_path()`` returns a remote URL and the log shows
    ``[omni.datastore] OmniHub is inaccessible``). The trimesh flat terrain is built in-process and
    a PreviewSurface visual material is procedural, so nothing is fetched. Physics material is
    preserved. Default (unset) behavior and the A3 path are unchanged.
    """
    import copy

    import isaaclab.sim as sim_utils
    import isaaclab.terrains as terrain_gen

    terrain = copy.deepcopy(scene.terrain)  # never mutate the shared MySceneCfg default
    terrain.terrain_type = "generator"
    terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
        curriculum=False,
        size=(8.0, 8.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
    )
    terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.30, 0.30))
    scene.terrain = terrain


@configclass
class CommandsCfg:
    """Motion imitation + racket target commands."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        anchor_body_name=G1_ANCHOR_BODY,
        body_names=G1_TRACKED_BODIES,
        motion_file=[FOREHAND_CLIP, BACKHAND_CLIP],  # clip 0 = forehand, clip 1 = backhand
        wrap_teleport=False,
        stand_start_prob=0.25,
        stand_start_min_hold=25,
        hold_steps_range=(0, 100),
        pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01),
                    "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
        velocity_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
                        "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)},
        joint_position_range=(-0.1, 0.1),
    )

    racket_target = mdp.RacketTargetCommandCfg(
        asset_name="robot",
        motion_command_name="motion",
        # 【中文】True = 在实际球拍接触点(racket_pos_w)画一个黑色小球，方便在 play/GUI 里看击球点。
        # 只在有渲染窗口时可见；headless 训练看不到但无害。不想看时改回 False 即可。
        debug_vis=True,
        # G1 racket mount FK (paddle on the right wrist). Set explicitly — the racket link may merge
        # into the wrist under USD import, so the FK falls back to (wrist pose) * (mount offset).
        # mount_offset / mount_quat are SEEDS from the URDF joint; tune in the viewer.
        wrist_body_name=G1_WRIST_BODY,
        racket_body_name=G1_RACKET_BODY,
        mount_offset=G1_MOUNT_OFFSET,
        mount_quat=G1_MOUNT_QUAT,
        mount_normal_axis=G1_MOUNT_NORMAL_AXIS,          # racket-local blade face
        mount_normal_sign_per_clip=(1.0, -1.0),          # forehand/backhand strike with opposite faces
        # TTRL frame: robot 0.23 m behind the near edge (pelvis at world x=-1.6, table centre at 0),
        # so the net lands at 0.23 + net_x(1.37) = 1.6 m in front of the station.
        table_near_x=0.23,
        strike_phase_per_clip=(0.47, 0.27),              # measured strike frames of the retargeted clips (FH ~0.47, BH ~0.27)
        strike_window_s=0.12,
        # STATION-RELATIVE racket target boxes, retuned to the retargeted G1 clips' MEASURED strike
        # apex (forehand ~(0.35,-0.08,0.97), backhand ~(0.18,-0.13,0.86)).
        # NOTE: both clips strike on the -y side -- verify the backhand clip is not mislabeled.
        racket_pos_range_per_clip=(
            ((0.28, 0.42), (-0.20, 0.05), (0.88, 1.05)),  # forehand
            ((0.10, 0.30), (-0.25, 0.00), (0.78, 0.95)),  # backhand
        ),
        racket_vel_range_per_clip=(
            ((1.0, 2.0), (0.5, 1.5), (0.2, 1.0)),    # forehand
            ((1.5, 2.5), (-1.5, -0.5), (0.0, 0.7)),  # backhand
        ),
        feet_body_names=tuple(G1_FEET_BODIES),
    )


@configclass
class ActionsCfg:
    """29-D clamped joint-position residual action (no passive joints)."""

    joint_pos = mdp.ClampedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
        passive_joint_names=G1_PASSIVE_HEAD_JOINT_NAMES,  # () — G1 has no neck
    )


@configclass
class ObservationsCfg:
    """105-D actor observation + privileged critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Order is fixed — it is the hope_pingpong_g1 observation contract.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = ObsTerm(func=mdp.applied_last_action, params={"action_name": "joint_pos"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_forward_xy = ObsTerm(
            func=mdp.base_forward_xy, params={"command_name": "racket_target"}, noise=Unoise(n_min=-0.02, n_max=0.02)
        )
        fixed_station_error_xy = ObsTerm(
            func=mdp.fixed_station_error_xy, params={"command_name": "racket_target"}, noise=Unoise(n_min=-0.03, n_max=0.03)
        )
        racket_target_rel_base = ObsTerm(
            func=mdp.racket_target_rel_base, params={"command_name": "racket_target"}, noise=Unoise(n_min=-0.02, n_max=0.02)
        )
        racket_target_vel_w = ObsTerm(func=mdp.racket_target_vel_w, params={"command_name": "racket_target"})
        time_to_strike = ObsTerm(func=mdp.time_to_strike, params={"command_name": "racket_target"})
        swing_side = ObsTerm(func=mdp.swing_side, params={"command_name": "racket_target"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # Actor terms (noise-free) ...
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.applied_last_action, params={"action_name": "joint_pos"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_forward_xy = ObsTerm(func=mdp.base_forward_xy, params={"command_name": "racket_target"})
        fixed_station_error_xy = ObsTerm(func=mdp.fixed_station_error_xy, params={"command_name": "racket_target"})
        racket_target_rel_base = ObsTerm(func=mdp.racket_target_rel_base, params={"command_name": "racket_target"})
        racket_target_vel_w = ObsTerm(func=mdp.racket_target_vel_w, params={"command_name": "racket_target"})
        time_to_strike = ObsTerm(func=mdp.time_to_strike, params={"command_name": "racket_target"})
        swing_side = ObsTerm(func=mdp.swing_side, params={"command_name": "racket_target"})
        # ... plus privileged (sim-only) signals for the value function.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        robot_body_pos_b = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        robot_body_ori_b = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        racket_pos_b = ObsTerm(func=mdp.racket_pos_b, params={"command_name": "racket_target"})
        racket_lin_vel_w = ObsTerm(func=mdp.racket_lin_vel_w, params={"command_name": "racket_target"})
        racket_normal_w = ObsTerm(func=mdp.racket_normal_w, params={"command_name": "racket_target"})
        racket_target_normal_w = ObsTerm(func=mdp.racket_target_normal_w, params={"command_name": "racket_target"})
        episode_time_left = ObsTerm(func=mdp.episode_time_left)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """The eleven reward terms. Weights/stds are illustrative examples — tune them."""

    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    imitation = RewTerm(
        func=mdp.sample_imitation,
        weight=1.0,
        params={"command_name": "motion", "std_pos": 0.3, "std_ori": 0.4, "body_names": G1_UPPER_TRACKED},
    )
    racket_position = RewTerm(
        func=mdp.racket_position, weight=4.0, params={"command_name": "racket_target", "std": 0.12}
    )
    racket_velocity = RewTerm(
        func=mdp.racket_velocity, weight=2.0, params={"command_name": "racket_target", "std": 0.6}
    )
    blade_direction = RewTerm(
        func=mdp.racket_blade_direction, weight=1.0, params={"command_name": "racket_target", "std": 0.3}
    )
    ball_contact = RewTerm(func=mdp.ball_contact, weight=2.0, params={"command_name": "racket_target"})
    net_cross = RewTerm(func=mdp.ball_net_cross, weight=2.0, params={"command_name": "racket_target"})
    opponent_bounce = RewTerm(func=mdp.ball_opponent_bounce, weight=4.0, params={"command_name": "racket_target"})
    follow_through_recovery = RewTerm(
        func=mdp.follow_through_recovery,
        weight=1.0,
        params={"command_name": "racket_target", "std": 0.5, "station_std": 0.3},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )


@configclass
class TerminationsCfg:
    """Time-out and physical-fall resets (ordinary env lifecycle, not a gate)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_tilted = DoneTerm(func=mdp.base_tilted, params={"threshold": 0.8})
    base_too_low = DoneTerm(func=mdp.base_too_low, params={"min_height": 0.5})


@configclass
class EventCfg:
    """Light domain randomization for sim-to-real robustness."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_ANCHOR_BODY),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )
    pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class G1HOPEPingPongEnvCfg(ManagerBasedRLEnvCfg):
    """The G1 HOPE task (gym id ``HOPE-PingPong-UnitreeG1-v0``)."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # Offline fallback: if the Isaac asset server is unreachable, swap the plane terrain (which
        # downloads a ground USD) for a procedural flat terrain. Opt-in via HOPE_OFFLINE_GROUND=1.
        # Applied before reading terrain.physics_material below (the swap preserves it).
        if os.environ.get("HOPE_OFFLINE_GROUND"):
            _apply_offline_ground(self.scene)

        # 50 Hz control (decimation 4 over a 200 Hz physics step).
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Robot + shared action adapter. default_q / action_scale / joint clamp all come from the
        # ONE shared G1 config the deploy runner reads (g1_deploy/.../action_adapter.yaml).
        adapter = load_g1_action_adapter_config()
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = adapter.default_q_by_name()
        self.actions.joint_pos.scale = adapter.action_scale_by_name()
        self.actions.joint_pos.position_clamp = adapter.position_clamp_by_name()

        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
