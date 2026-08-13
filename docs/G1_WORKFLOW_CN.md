# G1 乒乓全身控制工作流教程

将 Unitree **G1**(29 自由度)的正手/反手动捕挥拍数据，一路做成"训练好、已导出、仿真验证过"的乒乓策略的端到端教程。它是 A3 流程的 G1 版本，额外多了 G1 数据需要的动作预处理步骤。

```
动捕/重定向 CSV
   │  csv_to_npz_with_Interpolation.py   (Isaac 回放, --drop_leading 1)
   ▼
原始 npz  (37 个 body, Isaac 顺序, 内嵌 body_names)   ./motions/g1_*.npz
   │  convert_g1_full_motion.py          (按名字映射到 14 个跟踪 body + 重定向到 +x@原点)
   ▼
HOPE 格式 npz  (14 body, 29 关节, fps)   hope_training/motions/preprocessed/hope_g1_*.npz
   │  train.py task=HOPEPingPongG1
   ▼
checkpoint  →  export_onnx.py  →  ONNX  →  mujoco_eval_onnx.py --robot g1  (sim2sim)
```

> **最耗时间的两个坑**(务必先读):
> 1. **重定向(re-frame)必须真正写进你训练用的那个 `.npz` 里。** G1 数据采集时朝向 −x、位于 x≈1.6 m;
>    而乒乓任务假设机器人站在**原点、朝向 +x**。每次开跑前先跑一遍 [校验关卡](#3-校验关卡不可跳过)。
> 2. **预处理 `.npz` 与 git 是分开同步的。** 多机情况下,只提交转换脚本代码是**不够**的 ——
>    必须在"真正跑训练的那台机器"上重新生成(或就地重定向)这两个 `.npz`。

---

## 0. 前置条件

- 装好 **Isaac Lab 2.1**、`onnxruntime`、`mujoco`、`numpy` 的 conda 环境(仓库里的 `lab2.1`)。
- 环境变量(在任何会启动 Isaac 的命令之前设置 —— `csv_to_npz`、`train`、`export_onnx`):

  ```bash
  export HOPE_G1_USD_PATH="/绝对路径/到/g1_with_racket_adapter_short _ball_throwing/g1_with_racket_adapter_short _ball_throwing.usd"
  export HOPE_OFFLINE_GROUND=1   # 可选:程序化生成地面,给连不上 Isaac 资源服务器的机器用
  ```

- **输入**:两个重定向好的 G1 挥拍 CSV,例如 `motions/hope_g1_forehand.csv`、
  `motions/hope_g1_backhand.csv`。每行 36 列(通常 30 fps):

  | 第 0–2 列 | 第 3–6 列 | 第 7–35 列 |
  |---|---|---|
  | base 位置 xyz | base 四元数 **xyzw** | 29 个自由度(G1 标准关节顺序) |

  有些重定向工具会在第 0 行插入一个固定的参考/默认帧(原点、近似单位姿态)—— 这一行就是下面
  `--drop_leading 1` 要删掉的。

---

## 1. CSV → 原始 NPZ(Isaac 回放)

用真正的训练机器人(`G1_CFG`)回放 CSV 并记录全身世界姿态。每个 clip 跑一次:

```bash
python hope_training/whole_body_tracking/scripts/csv_to_npz_with_Interpolation.py \
    --input_file motions/hope_g1_forehand.csv --input_fps 180 --output_fps 50 \
    --output_name g1_forehand --drop_leading 1 --headless
python hope_training/whole_body_tracking/scripts/csv_to_npz_with_Interpolation.py \
    --input_file motions/hope_g1_backhand.csv --input_fps 180 --output_fps 50 \
    --output_name g1_backhand --drop_leading 1 --headless
```

输出 `./motions/g1_forehand.npz` / `g1_backhand.npz` —— **37 个 body**(Isaac 顺序)、`joint_pos`
29 维,并内嵌 `body_names` / `joint_names`(供下一步按名字映射)。

- `--drop_leading 1` 删掉第 0 行的伪参考帧(否则 clip 会在第 ~14 帧瞬移 ~1.6 m)。
- 起手的过渡段会把 base 固定在原地、只让关节渐入(不会从原点滑行)。
- 使用 `G1_CFG`(和训练同一个 USD),所以导出的 body 姿态与训练机器人一致。

## 2. 原始 NPZ → HOPE 14-body 格式(+ 重定向)

**按名字**挑出 14 个跟踪 body、把关节重排成标准顺序、把 `fps` 写成浮点,并把 clip 重定向为
"击球帧的 pelvis 位于**原点、朝向 +x**":

```bash
python hope_training/whole_body_tracking/scripts/convert_g1_full_motion.py \
    --in ./motions/g1_forehand.npz --out hope_training/motions/preprocessed/hope_g1_forehand.npz
python hope_training/whole_body_tracking/scripts/convert_g1_full_motion.py \
    --in ./motions/g1_backhand.npz --out hope_training/motions/preprocessed/hope_g1_backhand.npz
```

每次都应打印一行 `reframed: strike-frame pelvis -> origin +x`。可选参数:
`--no-reframe`(保留采集时的原始坐标系)、`--strike-phase 0.5`(把哪一帧放到 原点+x)。

**如果训练机上的转换脚本是旧版、没有重定向功能**(不打印 `reframed:` 行),就改用
[附录 A](#附录-a独立的就地重定向) 的脚本对现有 `.npz` 就地变换。

## 3. 校验关卡(不可跳过)

每次长训练前,确认训练将要加载的 clip 确实已经重定向:

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

期望输出:`pelvis [0. 0.] yaw 0 -> OK`。若显示 `pelvis [1.6 ...] yaw -180`,说明重定向没生效 ——
先修好再开跑,别浪费一次训练。

## 4. 训练

```bash
python hope_training/whole_body_tracking/scripts/train.py task=HOPEPingPongG1 headless=true
```

在 TensorBoard(`logs/rsl_rl/hope_pingpong_g1/<run>/`)里看。头几百个 iteration 的健康信号:

| 指标 | 坐标系错误(坏) | 重定向后(好) |
|---|---|---|
| `Metrics/racket_target/racket_pos_error` | 爬升到 ~0.5 | 下降趋向 ~0.2 |
| `Episode_Reward/ball_contact` | 恒为 **0** | 变成**非零** |
| `Metrics/racket_target/return_success` | 0 | 开始上升 |
| `Train/mean_episode_length` | 增长(站稳+挥拍) | 增长(不变) |
| `Episode_Termination/base_tilted` | → ~0 | → ~0 |

如果 `racket_pos_error` 仍卡在 0.5 附近,说明 clip 没被重定向(回到第 3 步)。

## 5. 导出 ONNX

```bash
python hope_training/whole_body_tracking/scripts/export_onnx.py \
    --task HOPE-PingPong-UnitreeG1-v0 \
    --checkpoint logs/rsl_rl/hope_pingpong_g1/<run>/model_<N>.pt \
    --onnx-name hope_pingpong_g1.onnx \
    --motion-file   hope_training/motions/preprocessed/hope_g1_forehand.npz \
    --motion-file-2 hope_training/motions/preprocessed/hope_g1_backhand.npz
```

- 需要设置 `HOPE_G1_USD_PATH`。生成 `<run>/exported/hope_pingpong_g1.onnx`(105→29)和一个 manifest。
- 有一道硬校验:确认关节顺序等于 G1 标准顺序,并把该顺序写进 ONNX 元数据。
- 导出器走 legacy 路径(`dynamo=False`)以得到干净的 **opset-11** 模型。若在很新的 torch 上
  `dynamo=False` 报错,改为在 `utils/exporter.py` 里把 `opset_version` 设为 18。

## 6. Sim2sim(MuJoCo)

先生成一次 G1 的 MuJoCo 场景(**要和源 MJCF 放在同一目录** —— 它用的是相对 `meshdir`):

```bash
python hope_training/whole_body_tracking/scripts/make_g1_mujoco_scene.py
# -> 在你的 Beyondmimic mjmodel.xml 旁边生成 g1_pingpong.xml
```

跑与部署一致的评测 —— **必须加 `--robot g1`**(默认是 A3 → 会报 111 维不匹配):

```bash
python hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py \
    --robot g1 \
    --onnx logs/rsl_rl/hope_pingpong_g1/<run>/exported/hope_pingpong_g1.onnx \
    --model-xml ~/Python_project/TTRL-ICRA2026/Beyondmimic_Deploy_G1/g1_pingpong.xml
```

会打印 `serves=… returns=… success_rate=…`。用 MuJoCo 查看器加载一次该 XML,把 `right_racket`
site 调到球拍中心。

## 7. 调参(修好之后)

重定向解决的是"大方向"的几何错位,但挥拍最高点仍可能落在采样目标框外。实测重定向后的最高点:
正手 ≈ (0.35, −0.08, 0.97),反手 ≈ (0.18, −0.13, 0.86)。当前目标框 `racket_pos_range_per_clip`
中心在 x≈0.5(正手 y<0,反手 y>0)。如果击球仍只是"部分成功":

- 在 `.../tasks/tracking/config/unitree_g1/hope_env_cfg.py` 里把 `racket_pos_range_per_clip`
  调到挥拍能到达的区域。
- 重点核对**反手的 y 侧**:挥拍把球拍留在略微 −y 一侧,而目标框却要 +y —— 确认正/反手 clip 没被
  搞反,并且目标回球方向与预期一致。

---

## 常见问题排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `stores 30/37 bodies but the task tracks 14` | 把全身原始 dump 直接喂给训练 | 跑第 2 步(`convert_g1_full_motion.py`) |
| clip 在第 ~14 帧瞬移 ~1.6 m、速度爆炸 | CSV 第 0 行是伪参考帧 | 第 1 步加 `--drop_leading 1` |
| 训练正常但 `return_success=0`、`racket_pos_error→0.5` | clip 朝向 −x / 有偏移 | 重定向(第 2 步)+ 校验(第 3 步) |
| "新"训练结果和旧的逐位相同 | 训练用的还是同一批旧坐标系 `.npz` | 在**训练机**上重新生成/重定向 clip,并校验 |
| `observation input trailing dim 105 != expected 111` | 评测默认走了 A3 包 | 加 `--robot g1` |
| ONNX 导出打印 `CastLike` 版本转换 traceback | dynamo/opset-18 图往 opset-11 降版失败 | 无害(文件已写出);`dynamo=False` / opset 18 可消除 |
| `input MJCF not found` / 缺 `g1_pingpong.xml` | 场景没生成 | 跑 `make_g1_mujoco_scene.py` |

## 附录 A:独立的就地重定向

当训练机上的 `convert_g1_full_motion.py` 是重定向之前的旧版(不打印 `reframed:` 行)时用这个。
它直接对你已有的预处理 `.npz` 做变换 —— 不依赖 git、不新建文件。可重复运行(已重定向的 clip 再跑
是空操作)。在仓库根目录运行,然后重跑 [校验关卡](#3-校验关卡不可跳过):

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

## 参考

- G1:**29 自由度**,观测 **105**(= 18 + 3·29),动作 29,**14 个跟踪 body**,锚点 `torso_link`,根 `pelvis`。
- 标准关节顺序:`hope_training/config/joint_order_unitree_g1.yaml`。
- 任务:训练用 hydra `task=HOPEPingPongG1`;gym id 为 `HOPE-PingPong-UnitreeG1-v0`。
- 占位(非性能)clip 可用 `make_g1_placeholder_motions.py` 重新生成。
