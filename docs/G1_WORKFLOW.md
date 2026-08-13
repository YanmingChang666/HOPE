# G1 Ping-Pong Whole-Body Workflow

End-to-end guide for taking a Unitree **G1** (29-DOF) forehand/backhand mocap swing all the way to a
trained, exported, sim-verified ping-pong policy. This is the G1 twin of the A3 pipeline, with the
extra motion-preprocessing steps the G1 clips need.

```
mocap/retarget CSV
   │  csv_to_npz_with_Interpolation.py   (Isaac replay, --drop_leading 1)
   ▼
raw npz  (37 bodies, Isaac order, body_names embedded)   ./motions/g1_*.npz
   │  convert_g1_full_motion.py          (name-map to 14 tracked bodies + re-frame to +x@origin)
   ▼
HOPE-schema npz  (14 bodies, 29 joints, fps)   hope_training/motions/preprocessed/hope_g1_*.npz
   │  train.py task=HOPEPingPongG1
   ▼
checkpoint  →  export_onnx.py  →  ONNX  →  mujoco_eval_onnx.py --robot g1  (sim2sim)
```

> **The two mistakes that cost the most time** (read these first):
> 1. **Re-framing must be in the `.npz` you actually train on.** G1 clips are captured facing −x at
>    x≈1.6 m; the ping-pong task assumes the robot at the origin facing **+x**. Always run the
>    [verify gate](#3-verify-gate--do-not-skip) before a run.
> 2. **The preprocessed `.npz` sync separately from git.** On a multi-machine setup, committing the
>    converter is **not** enough — regenerate (or re-frame) the clips on the machine that trains.

---

## 0. Prerequisites

- Conda env with **Isaac Lab 2.1**, `onnxruntime`, `mujoco`, `numpy` (the repo's `lab2.1`).
- Environment variables (set before any Isaac-launching command — `csv_to_npz`, `train`, `export_onnx`):

  ```bash
  export HOPE_G1_USD_PATH="/abs/path/to/g1_with_racket_adapter_short _ball_throwing/g1_with_racket_adapter_short _ball_throwing.usd"
  export HOPE_OFFLINE_GROUND=1   # optional: procedural ground, for machines that can't reach the Isaac asset server
  ```

- **Input**: two retargeted G1 swing CSVs, e.g. `motions/hope_g1_forehand.csv`,
  `motions/hope_g1_backhand.csv`. Row layout (36 columns, 180 fps for this mocap — pass the real rate via `--input_fps`):

  | cols 0–2 | cols 3–6 | cols 7–35 |
  |---|---|---|
  | base position xyz | base quaternion **xyzw** | 29 DOF (canonical G1 joint order) |

  Some retargeters prepend a fixed reference/default row (origin, identity-ish) as row 0 — that is
  the row `--drop_leading 1` removes below.

---

## 1. CSV → raw NPZ (Isaac replay)

Replays the CSV on the real training robot (`G1_CFG`) and logs full-body world poses. Run once per clip:

```bash
python hope_training/whole_body_tracking/scripts/csv_to_npz_with_Interpolation.py \
    --input_file motions/hope_g1_forehand.csv --input_fps 180 --output_fps 50 \
    --output_name g1_forehand --drop_leading 1 --headless
python hope_training/whole_body_tracking/scripts/csv_to_npz_with_Interpolation.py \
    --input_file motions/hope_g1_backhand.csv --input_fps 180 --output_fps 50 \
    --output_name g1_backhand --drop_leading 1 --headless
```

Writes `./motions/g1_forehand.npz` / `g1_backhand.npz` — **37 bodies** in Isaac order, `joint_pos`
29-wide, plus `body_names` / `joint_names` (so the next step can map by name).

- `--drop_leading 1` strips the bogus reference row 0 (otherwise the clip teleports ~1.6 m at frame ~14).
- The head "settle-in" lead-in holds the base in place and only eases the joints (no origin glide).
- Uses `G1_CFG` (same USD as training) so exported body poses match the trained robot.

## 2. Raw NPZ → HOPE 14-body schema (+ re-frame)

Selects the 14 tracked bodies **by name**, reorders joints to the canonical order, writes `fps` as a
float, and re-frames the clip so the strike-frame pelvis sits at the **origin facing +x**:

```bash
python hope_training/whole_body_tracking/scripts/convert_g1_full_motion.py \
    --in ./motions/g1_forehand.npz --out hope_training/motions/preprocessed/hope_g1_forehand.npz
python hope_training/whole_body_tracking/scripts/convert_g1_full_motion.py \
    --in ./motions/g1_backhand.npz --out hope_training/motions/preprocessed/hope_g1_backhand.npz
```

Each run must print a `reframed: strike-frame pelvis -> origin +x` line. Flags:
`--no-reframe` (keep the captured frame), `--strike-phase 0.5` (which frame is placed at origin+x).

**If the training machine's converter is older and has no re-framing** (no `reframed:` line), transform
the existing `.npz` in place instead — see [Appendix A](#appendix-a-standalone-in-place-re-frame).

## 3. VERIFY GATE — do not skip

Before every long run, confirm the clips training will load are actually re-framed:

```bash
python3 - <<'EOF'
import numpy as np
yaw=lambda q:np.degrees(np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2)))
for f in ["forehand","backhand"]:
    d=np.load(f"hope_training/motions/preprocessed/hope_g1_{f}.npz"); bp=d["body_pos_w"]; s=len(bp)//2
    print(f,"pelvis",bp[s,0,:2].round(2),"yaw",round(yaw(d["body_quat_w"][s,0])),
          "-> OK" if abs(bp[s,0,0])<0.3 else "-> STILL OLD FRAME — STOP")
EOF
```

Expected: `pelvis [0. 0.] yaw 0 -> OK`. If it shows `pelvis [1.6 ...] yaw -180`, the re-frame did not
take — fix it before wasting a run.

## 4. Train

```bash
python hope_training/whole_body_tracking/scripts/train.py task=HOPEPingPongG1 headless=true
```

Monitor in TensorBoard (`logs/rsl_rl/hope_pingpong_g1/<run>/`). Health signals in the first few
hundred iterations:

| tag | mis-framed (bad) | re-framed (good) |
|---|---|---|
| `Metrics/racket_target/racket_pos_error` | climbs to ~0.5 | falls toward ~0.2 |
| `Episode_Reward/ball_contact` | flat **0** | becomes **nonzero** |
| `Metrics/racket_target/return_success` | 0 | starts rising |
| `Train/mean_episode_length` | grows (stand+swing) | grows (unchanged) |
| `Episode_Termination/base_tilted` | → ~0 | → ~0 |

If `racket_pos_error` is still stuck near 0.5, the clips were not re-framed (go back to step 3).

## 5. Export ONNX

```bash
python hope_training/whole_body_tracking/scripts/export_onnx.py \
    --task HOPE-PingPong-UnitreeG1-v0 \
    --checkpoint logs/rsl_rl/hope_pingpong_g1/<run>/model_<N>.pt \
    --onnx-name hope_pingpong_g1.onnx \
    --motion-file   hope_training/motions/preprocessed/hope_g1_forehand.npz \
    --motion-file-2 hope_training/motions/preprocessed/hope_g1_backhand.npz
```

- Needs `HOPE_G1_USD_PATH` set. Produces `<run>/exported/hope_pingpong_g1.onnx` (105→29) + a manifest.
- A hard gate verifies the articulation joint order equals the canonical G1 order and stamps it into
  the ONNX metadata.
- The exporter uses the legacy path (`dynamo=False`) for a clean **opset-11** model. On very new
  torch, if `dynamo=False` errors, bump `opset_version` to 18 in `utils/exporter.py` instead.

## 6. Sim2sim (MuJoCo)

Generate the G1 MuJoCo scene once (keep it **next to** the source MJCF — it uses a relative `meshdir`):

```bash
python hope_training/whole_body_tracking/scripts/make_g1_mujoco_scene.py
# -> writes g1_pingpong.xml next to your Beyondmimic mjmodel.xml
```

Run the deploy-faithful eval — **`--robot g1` is required** (default is A3 → 111-D obs mismatch):

```bash
python hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py \
    --robot g1 \
    --onnx logs/rsl_rl/hope_pingpong_g1/<run>/exported/hope_pingpong_g1.onnx \
    --model-xml ~/Python_project/TTRL-ICRA2026/Beyondmimic_Deploy_G1/g1_pingpong.xml
```

Prints `serves=… returns=… success_rate=…`. Load the XML once in the MuJoCo viewer and tune the
`right_racket` site to the paddle centre.

## 7. Tuning (after the fix lands)

Re-framing fixes the gross geometry, but the swing apex may still miss the sampled target box. Measured
re-framed apex: forehand ≈ (0.35, −0.08, 0.97), backhand ≈ (0.18, −0.13, 0.86). Current box
`racket_pos_range_per_clip` centres near x≈0.5 (forehand y<0, backhand y>0). If contacts stay partial:

- Retune `racket_pos_range_per_clip` in
  `.../tasks/tracking/config/unitree_g1/hope_env_cfg.py` to the swing's reachable zone.
- Sanity-check the **backhand y-side**: the swing keeps the paddle slightly −y while the box expects
  +y — confirm your forehand/backhand clips aren't swapped and the intended return side matches.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `stores 30/37 bodies but the task tracks 14` | raw full-body dump fed to training | run step 2 (`convert_g1_full_motion.py`) |
| clip teleports ~1.6 m at frame ~14; huge velocities | bogus reference row 0 in CSV | `--drop_leading 1` in step 1 |
| trains fine but `return_success=0`, `racket_pos_error→0.5` | clips face −x / offset | re-frame (step 2) + verify (step 3) |
| a "new" run is bit-identical to the old one | trained on the same old-frame `.npz` | regenerate/re-frame clips **on the training machine**, verify |
| `observation input trailing dim 105 != expected 111` | eval defaulted to A3 package | add `--robot g1` |
| ONNX export prints a `CastLike` version-convert traceback | opset-11 down-convert of a dynamo/opset-18 graph | harmless (file still written); `dynamo=False` / opset 18 removes it |
| `input MJCF not found` / missing `g1_pingpong.xml` | scene not generated | run `make_g1_mujoco_scene.py` |

## Appendix A: standalone in-place re-frame

Use this when the training machine's `convert_g1_full_motion.py` predates re-framing (no `reframed:`
line). It transforms the preprocessed `.npz` you already have — no git, no new files. Safe to re-run
(already-reframed clips are a no-op). Run from the repo root, then re-run the [verify gate](#3-verify-gate--do-not-skip):

```bash
python3 - <<'EOF'
import numpy as np
def yaw(q): w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def qmul(a,b):
    aw,ax,ay,az=a[...,0],a[...,1],a[...,2],a[...,3]; bw,bx,by,bz=b[...,0],b[...,1],b[...,2],b[...,3]
    return np.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw],-1)
for f in ["forehand","backhand"]:
    p=f"hope_training/motions/preprocessed/hope_g1_{f}.npz"; d=dict(np.load(p))
    bp,bq=d["body_pos_w"],d["body_quat_w"]; s=len(bp)//2
    th=-float(yaw(bq[s,0])); cz,sz=np.cos(th),np.sin(th)
    Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]); qz=np.array([np.cos(th/2),0,0,np.sin(th/2)])
    c=bp[s,0].astype(float).copy(); c[2]=0.0
    d["body_pos_w"]=((bp-c)@Rz.T).astype(np.float32)
    d["body_quat_w"]=qmul(np.broadcast_to(qz,bq.shape),bq).astype(np.float32)
    d["body_lin_vel_w"]=(d["body_lin_vel_w"]@Rz.T).astype(np.float32)
    d["body_ang_vel_w"]=(d["body_ang_vel_w"]@Rz.T).astype(np.float32)
    np.savez(p,**d)
    b=d["body_pos_w"]; print(f,"-> pelvis",b[s,0,:2].round(2),"yaw",round(np.degrees(yaw(d['body_quat_w'][s,0]))),"OK" if abs(b[s,0,0])<0.3 else "FAIL")
EOF
```

## Reference

- G1: **29 DOF**, obs **105** (= 18 + 3·29), action 29, **14 tracked bodies**, anchor `torso_link`, root `pelvis`.
- Canonical joint order: `hope_training/config/joint_order_unitree_g1.yaml`.
- Task: train hydra `task=HOPEPingPongG1`; gym id `HOPE-PingPong-UnitreeG1-v0`.
- Placeholders (non-performant) can be regenerated with `make_g1_placeholder_motions.py`.
