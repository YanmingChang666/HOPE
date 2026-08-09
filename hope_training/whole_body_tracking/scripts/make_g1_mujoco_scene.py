#!/usr/bin/env python3
"""Turn a working G1 ``g1_with_racket`` MuJoCo model into the ``g1_pingpong.xml`` the HOPE
MuJoCo evaluator / deploy runner expect.

The HOPE MuJoCo scene (``scripts/mujoco_pingpong_scene.py``) and the deploy sim bridge load a robot
MJCF and look up a few elements BY NAME:

  * ``pelvis_free_joint``     — the floating base joint,
  * ``pelvis_imu_gyro``       — a gyro (base angular velocity) sensor,
  * ``right_racket_collision``— the racket collision geom (ball<->paddle contact pair),
  * ``right_racket``          — a site at the paddle (racket pose readout),
  * the 29 G1 joints + torque actuators.

Your ``Beyondmimic_Deploy_G1/mjmodel.xml`` (``g1_with_racket``) already has the joints, actuators, a
free base joint (``floating_base_joint``), IMU sensors, and the racket body — it just uses different
names and has no racket site. This script copies it and applies the minimal renames/additions, so the
mesh paths (``meshdir``) stay valid **when the output is written next to the input**. It does NOT
touch physics/joint values.

NO MuJoCo dependency (pure stdlib XML) — but you should load the result once in MuJoCo to confirm.

Usage:
    python3 scripts/make_g1_mujoco_scene.py \
        --in  ~/Python_project/TTRL-ICRA2026/Beyondmimic_Deploy_G1/mjmodel.xml \
        --out ~/Python_project/TTRL-ICRA2026/Beyondmimic_Deploy_G1/g1_pingpong.xml
"""

from __future__ import annotations

import argparse
import pathlib
import xml.etree.ElementTree as ET

_DEFAULT_IN = "/home/cym/Python_project/TTRL-ICRA2026/Beyondmimic_Deploy_G1/mjmodel.xml"

# Names the HOPE MuJoCo scene / deploy bridge look up.
FREE_JOINT_NAME = "pelvis_free_joint"
GYRO_SENSOR_NAME = "pelvis_imu_gyro"
RACKET_COLLISION_GEOM = "right_racket_collision"
RACKET_SITE = "right_racket"
RACKET_BODY = "table_tennis_racket_link"


def _rename_free_joint(root: ET.Element) -> str:
    """Rename the base free joint (``<freejoint>`` or ``<joint type='free'>``) to pelvis_free_joint."""
    for parent in root.iter():
        for el in list(parent):
            if el.tag == "freejoint":
                el.set("name", FREE_JOINT_NAME)
                return "freejoint"
            if el.tag == "joint" and el.get("type") == "free":
                el.set("name", FREE_JOINT_NAME)
                return f"joint '{el.get('name')}'"
    raise SystemExit("no free/floating base joint found in the input MJCF")


def _fix_racket(root: ET.Element) -> list[str]:
    """Rename the racket collision geom and add a racket site on the racket body."""
    notes = []
    for body in root.iter("body"):
        if body.get("name") != RACKET_BODY:
            continue
        # Collision geom = a geom that is NOT visual-only (group 1 / contype=0 & conaffinity=0).
        col = None
        for g in body.findall("geom"):
            grp, ct, ca = g.get("group"), g.get("contype"), g.get("conaffinity")
            is_visual = grp == "1" or (ct == "0" and ca == "0")
            if not is_visual:
                col = g
                break
        if col is None:  # fall back to a name match
            col = next((g for g in body.findall("geom") if "col" in (g.get("name") or "")), None)
        if col is None:
            raise SystemExit(f"no collision geom found on body '{RACKET_BODY}'")
        col.set("name", RACKET_COLLISION_GEOM)
        notes.append(f"racket collision geom -> {RACKET_COLLISION_GEOM}")
        # Add the racket site at the body origin (paddle mount). TUNE its pos to the paddle centre.
        if body.find(f"site[@name='{RACKET_SITE}']") is None:
            site = ET.SubElement(body, "site")
            site.set("name", RACKET_SITE)
            site.set("pos", "0 0 0")   # SEED — move to the paddle blade centre in the MuJoCo viewer
            site.set("size", "0.01")
            site.set("group", "3")
            notes.append(f"added site '{RACKET_SITE}' at racket body origin (TUNE pos)")
        return notes
    raise SystemExit(f"racket body '{RACKET_BODY}' not found in the input MJCF")


def _add_gyro(root: ET.Element) -> str:
    """Ensure a ``pelvis_imu_gyro`` frameangvel sensor on the pelvis body exists."""
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    if sensor.find(f"*[@name='{GYRO_SENSOR_NAME}']") is not None:
        return f"gyro '{GYRO_SENSOR_NAME}' already present"
    fa = ET.SubElement(sensor, "frameangvel")
    fa.set("objtype", "body")
    fa.set("objname", "pelvis")
    fa.set("name", GYRO_SENSOR_NAME)
    return f"added gyro sensor '{GYRO_SENSOR_NAME}' (frameangvel on pelvis)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_xml", default=_DEFAULT_IN, help="Source g1_with_racket MJCF.")
    ap.add_argument("--out", dest="out_xml", default=None,
                    help="Output g1_pingpong.xml (default: g1_pingpong.xml next to the input, so meshdir resolves).")
    args = ap.parse_args()

    in_path = pathlib.Path(args.in_xml).expanduser()
    if not in_path.is_file():
        raise SystemExit(f"input MJCF not found: {in_path}")
    out_path = pathlib.Path(args.out_xml).expanduser() if args.out_xml else in_path.with_name("g1_pingpong.xml")

    tree = ET.parse(in_path)
    root = tree.getroot()
    root.set("model", "g1_pingpong")

    notes = [f"free joint -> {FREE_JOINT_NAME} (was {_rename_free_joint(root)})"]
    notes += _fix_racket(root)
    notes.append(_add_gyro(root))

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"[make_g1_mujoco_scene] wrote {out_path}")
    for n in notes:
        print(f"  - {n}")
    print("  NOTE: meshdir is relative — keep this file next to the source (or fix meshdir). "
          "Load it once in MuJoCo to confirm, and tune the 'right_racket' site to the paddle centre.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
