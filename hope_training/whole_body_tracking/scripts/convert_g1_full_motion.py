#!/usr/bin/env python3
"""Convert a raw BeyondMimic G1 export into the HOPE 14-tracked-body motion schema.

The BeyondMimic ``csv_to_npz`` pipeline logs ``robot.data.joint_pos`` / ``body_pos_w`` verbatim,
i.e. ALL ~30 articulation bodies and the DOFs in that robot's own Isaac order. The HOPE task, on the
other hand, imitates only the 14 tracked bodies (``G1_TRACKED_BODIES``) and expects the joint axis in
HOPE's canonical order (``G1_JOINT_NAMES``). Feeding a raw 30-body export straight into training
fails the ``MotionLoader`` body-count check (``stores 30 bodies but the task tracks 14``).

This script maps the raw export into the HOPE schema **by name**, so it is correct even when the
export robot's ordering differs from HOPE's:

    * selects the 14 ``G1_TRACKED_BODIES`` from the full body axis, in that exact order;
    * reorders the joint axis from the export order into ``G1_JOINT_NAMES``;
    * writes ``fps`` as a float32 scalar.

It requires the export to carry its own ``body_names`` / ``joint_names`` arrays (add the two
``log[...] = np.array(robot.*_names)`` lines to ``csv_to_npz_with_Interpolation.py`` and re-export).
Mapping by name is the whole point — without the names we cannot know the raw axis order, so the
script refuses to guess.

Dependency-light (numpy only, NO Isaac Lab); the canonical name lists are imported from the sibling
``make_g1_placeholder_motions.py`` so they stay in lockstep with ``robots/g1.py``.

Usage (from the repo root):
    python hope_training/whole_body_tracking/scripts/convert_g1_full_motion.py \
        --in  /path/to/raw_forehand.npz \
        --out hope_training/motions/preprocessed/hope_g1_forehand.npz
    # ...and again for the backhand clip.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

# Reuse the authoritative name lists (numpy/xml-only module — safe to import, no Isaac).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_g1_placeholder_motions import G1_JOINT_NAMES, G1_TRACKED_BODIES  # noqa: E402

_ARRAY_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def _as_str_list(arr) -> list[str]:
    """np.load may hand back <U / bytes / object arrays — normalize to a list[str]."""
    out = []
    for v in np.asarray(arr).reshape(-1).tolist():
        out.append(v.decode() if isinstance(v, (bytes, bytearray)) else str(v))
    return out


def _index_map(target: list[str], source: list[str], what: str) -> list[int]:
    """For each name in ``target``, its position in ``source`` (errors on any missing name)."""
    pos = {name: i for i, name in enumerate(source)}
    missing = [n for n in target if n not in pos]
    if missing:
        raise ValueError(
            f"Export is missing {what} required by the HOPE schema: {missing}\n"
            f"  export {what}: {source}"
        )
    return [pos[n] for n in target]


def convert(src_path: str, dst_path: str) -> None:
    data = np.load(src_path, allow_pickle=True)

    for k in ("body_names", "joint_names"):
        if k not in data.files:
            raise ValueError(
                f"{src_path} has no '{k}'. Re-export with the ordering saved:\n"
                "  add to csv_to_npz_with_Interpolation.py before np.savez:\n"
                "    log['body_names']  = np.array(robot.body_names)\n"
                "    log['joint_names'] = np.array(robot.joint_names)\n"
                "then re-run the export. Mapping by name needs these; the script won't guess order."
            )
    src_joints = _as_str_list(data["joint_names"])
    src_bodies = _as_str_list(data["body_names"])

    joint_idx = _index_map(G1_JOINT_NAMES, src_joints, "joint names")
    body_idx = _index_map(G1_TRACKED_BODIES, src_bodies, "body names")

    jp, jv = data["joint_pos"], data["joint_vel"]
    if jp.shape[1] != len(src_joints):
        raise ValueError(f"joint_pos width {jp.shape[1]} != len(joint_names) {len(src_joints)} in {src_path}")
    bp = data["body_pos_w"]
    if bp.shape[1] != len(src_bodies):
        raise ValueError(f"body_pos_w bodies {bp.shape[1]} != len(body_names) {len(src_bodies)} in {src_path}")

    fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    out = {
        "fps": np.float32(fps),
        "joint_pos": jp[:, joint_idx].astype(np.float32),
        "joint_vel": data["joint_vel"][:, joint_idx].astype(np.float32),
        "body_pos_w": data["body_pos_w"][:, body_idx].astype(np.float32),
        "body_quat_w": data["body_quat_w"][:, body_idx].astype(np.float32),  # wxyz, unchanged
        "body_lin_vel_w": data["body_lin_vel_w"][:, body_idx].astype(np.float32),
        "body_ang_vel_w": data["body_ang_vel_w"][:, body_idx].astype(np.float32),
    }

    dst = pathlib.Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    print(
        f"[convert_g1_full_motion] {os.path.basename(src_path)} -> {dst}\n"
        f"    frames {out['joint_pos'].shape[0]}, joints {out['joint_pos'].shape[1]}, "
        f"bodies {out['body_pos_w'].shape[1]}, fps {fps:g}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="Raw BeyondMimic export .npz (30-body).")
    ap.add_argument("--out", dest="dst", required=True, help="HOPE-schema .npz to write (14-body).")
    args = ap.parse_args()
    if not os.path.isfile(args.src):
        raise FileNotFoundError(f"Input not found: {args.src}")
    convert(args.src, args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
