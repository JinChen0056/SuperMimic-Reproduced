# SuperMimic 复现审计与执行计划（2026-10-23）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + reproducibility audit
- Origin Date: 2026-10-23
- Verification Status: UNVERIFIED（论文结果尚未在本机复现）
- Version Label: reproduction_audit_v1
- 输入：论文原文、两个固定 commit 的源码；仓库 README 数字是上游自报，不是本次实验结果。

## 结论与范围

采用本仓库 mjlab 实现作为基线，`supermimic_vibed` 作为独立实现交叉检查。先复现 LAFAN1 40 clips / 2.45 h 的 Fig.4a，再扩展 BONES-SEED；本阶段不涉及真机部署。严格观测与增强观测必须分开命名、训练、评测，不能用增强版高分证明严格复现。

来源：[论文](https://arxiv.org/pdf/2610.28480)、[vibed](https://github.com/JinChen0056/supermimic_vibed)、[Reproduced](https://github.com/JinChen0056/SuperMimic-Reproduced)。源码快照：vibed `a1508d42c9d1699a52e8233ab16d08c488441ab9`；Reproduced `d9be385b3eeb13a638fb33fb200d3bfd9be9839f`。

## 论文要点与必须守住的细节

1. 在线交替 Collect → Fit WM → Improve policy；不是离线预训练完世界模型后用 PPO。旧策略数据对动力学仍有效，但数据必须包含实际执行的带噪动作及对应 sim context。
2. Eq.1：k=8 自回归展开，不能 teacher forcing。世界模型预测状态，奖励/观测/终止允许解析计算；学习的 head 才产生相应监督损失。
3. Eq.2：冻结世界模型**参数**，保留对输入状态/动作的导数，H=32 BPTT；目标为 Σ γ^t c_t r_(t+1) + γ^H c_H V_H。不能在 WM 前向外包 no_grad。c_(t+1)=c_t(1-d_(t+1))。
4. Eq.3：critic 只用想象轨迹，GAE λ=.95，固定 targets，5 epochs × 4 minibatches，无 EMA target critic。bootstrap 项必须对 s_H 保留梯度。
5. Table A.1：每帧本体 93 维，10 帧历史；29 关节位置、29 速度、3 IMU 角速度、3 重力、29 上一动作。goal 是 10 帧未来参考。论文未充分指定 goal 的坐标表示、state 参数化、激活/归一化等，不能把实现选择当作论文事实。
6. Table A.2/A.3：BeyondMimic tracking 奖励去自碰撞惩罚、joint limit 权重 −2；终止为 pelvis 高度误差 >.25 m 或重力 z 分量差 >.8；摩擦、关节偏置、torso CoM 输入 WM，actor 不读取这些特权量。去除 self-collision penalty 不等于关闭 self-collision physics。
7. Table A.4/A.5：WM (2048,2048,2048)，encoder/decoder (2048,1024,512,256)，latent 32 tanh，critic (256,128)；4096 env，replay 每 env 1000，WM 每更新 10×2048，policy batch 4096，UTD .4，Adam 2e-4，γ=.99。
8. 1s bins 的 return-based reset：50% uniform、40% softmax(-25 G/Gmax)、10% 历史权重。必须检查窗口不跨 reset、历史动作对齐、timeout 与 terminal 分离。
9. Appendix F：每 clip 首帧完整播放，无 DR/观测噪声/动作噪声，摩擦 1；失败采用训练 termination。归一化回报用**完整动作**的最大可能累计回报，失败后未执行部分视为零。
10. Fig.4a 的 .82 / 1.2h 来自 H100；本机 RTX 4090 的时长不能当作同硬件效率复现。全量论文是 129,514 clips / 264.56h，不等同于上游筛选出的 129,943 / 263.75h。

## 两个实现的异同（源码核对）

| 项目 | supermimic_vibed | SuperMimic-Reproduced |
|---|---|---|
| 仿真集成 | Isaac Lab 3 kitless → Newton → MuJoCo Warp，Docker | mjlab 固定 8bb2bc9 → MuJoCo Warp，uv.lock |
| 工程组织 | 10 个核心模块，手写 wrapper；部署需模型 URDF→MJCF→USD | src 包、数值测试、独立 env / WM / imagination / benchmark，可直接使用 mjlab G1 |
| 方法主干 | 8步 WM，32步 BPTT，survival+critic | 同左，Eq.2/Eq.3 源码形式一致 |
| 状态 | 71维，旋转矩阵接口，局部增量 | 71维 MuJoCo 布局，速度增量+显式积分+位置校正 |
| actor 输入 | 每帧 99=93+6 参考相对朝向，进入历史；goal 73×10 | 默认 930 本体历史+680 goal；可选额外9维参考相对位置/朝向进入 goal |
| 终止 head | 解析阈值平滑 sigmoid | 学习 MLP(64,64)，BCE；可能在早期 WM 不准时干扰 survival |
| 精度 | 支持 bf16，README 成绩用 bf16；关闭自碰撞 | 需要继续核对实际 TF32/physics 配置，不假定与论文一致 |
| 预热/时钟 | 先64仿真步，额外500次 WM-only；iteration 为更新循环 | 40仿真步后即联合；iteration 为仿真步，.4次更新/步 |
| LR 时钟 | 250k更新循环衰减 | 250k仿真步衰减；同名 iterations 不可横向比较 |
| 数据规模 | LAFAN1，40 clips | LAFAN1 + BONES-SEED + 紧凑库/多卡 |
| 评测分母 | 完整 clip_len−1 | 上游用实际存活步数；本分支修复完整长度分母与聚合权重 |
| 自报成绩 | .829 / 0失败，RTX PRO 6000 1.47h | 增强观测 .904 / 3失败，5090 3.3h；论文观测 e2 .56 / 9失败 |

**当前判断**：第二套更适合可审计扩展，第一套有很好的独立对照价值。两套的高分不构成等价复现证据；尤其第二套历史分数需要用修正后的指标重评。93维本体输入无法直接感知全局 xy/yaw 漂移，能否仅靠历史与参考序列获得足够恢复能力需实验检验，不能据此直接断言论文错误。

## 发现与修正

- 已确认 benchmark.py 用 `ret_sum / (.1 * survived_steps)`，早失败仍可能高分；改为完整参考 transition 数，聚合也按完整长度加权；加定量回归测试（含模拟环境的尾步检查）。
- 已处理评测尾步：按完整参考的 L−1 个 transition 截止累计，自动 reset 前额外物理步不累计奖励或失败；增加完美跟踪后额外步跌倒的 mock 环境回归。真实环境尾步时序仍需集成验证。
- 待核验：world-model 误差的标准化尺度来自实现选择，两库的 NSE 不能直接与论文图上的数值等同。
- 待核验：sim context 关节偏置的物理/观测含义与符号、pelvis/torso anchor 的奖励对应、joint order、quat xyzw/wxyz、角速度 world/body 坐标。

## 分阶段规划与验收

| 阶段 | 执行内容 | 验收 | 预算/停止条件 |
|---|---|---|---|
| P0 环境与来源 | 锁代码/数据版本，uv frozen，设备/磁盘记录 | CUDA可用，依赖可导入，来源含SHA256 | 下载约数GB；磁盘保留10GB以上 |
| P1 数值正确性 | FK/velocity 对 MuJoCo、解析头对 sim、GAE、分母回归 | 测试通过；缺数据造成 skip 不能算通过 | 分开记录 passed/skipped/failed |
| P2 数据与闭环 | 先 walk/dance/jumps CSV→50Hz npz，再40条；小batch训练、保存/恢复、评测 | 有限loss/梯度、参数更新、checkpoint、评测JSON | 缩小环境/网络仅用于冒烟，不计论文成绩 |
| P3 LAFAN1严格基线 | 默认93维本体，完整网络/H=32/k=8；先小预算profile再完整4096env | 完整长度回报、失败数、每项奖励与追踪误差，记录GPU小时 | 首轮3h；不自动以H100时间作比较 |
| P4 有控制的诊断 | strict vs +6朝向 vs +9位姿；同seed/预算；解析 vs 学习终止；WM-only预热 | 分离信息增益、训练稳定性和效率；主配置至少3 seeds | 先单因素，未验证前不堆叠修改 |
| P5 全量与基线 | BONES筛选manifest对齐、PPO-Sim、PPO-WM、H×critic和push消融 | 同数据/奖励/观测/评测；分组失败和扩展曲线 | 当前35GB余量不启动全量下载；需独立存储预算 |

P3 启动前完成 P1/P2 并核对评测尾步。汇报 full-clip mean return、full-length weighted return、failed/40，同时报告 survived_steps；训练日志中的 survived-episode return 仅作诊断。在线 WM NSE 受策略分布影响，还需固定 held-out replay 的 1/8/32步误差。每次实验保存 config、seed、git SHA、数据manifest、依赖、实际设备、训练与编译计时；评测耗时单列。主实验失败须保留日志并分析，不能静默换观测后沿用原名称。

## 执行入口

工作目录为本仓库。下载脚本固定 HF revision，并对每条CSV写 SHA256 到被忽略的 data/raw/lafan1/source.json：

```bash
python3 scripts/reproduction/download_lafan1.py
env -u PYTHONPATH uv sync --frozen --group dev
env -u PYTHONPATH uv run python scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 --input-fps 30 --workers 4
env -u PYTHONPATH uv run pytest tests -q
```

完整基线命令（P1/P2及尾步审核通过后执行）：

```bash
env -u PYTHONPATH uv run sm-train --run-name lafan1_strict_s0 --seed 0 --schedule.max-hours 3
# 完成后使用修复的 benchmark；不和旧版分数混用。
env -u PYTHONPATH uv run python scripts/eval_benchmark.py --ckpt logs/lafan1_strict_s0/ckpt_latest.pt --npz-dir data/npz/lafan1
```

运行过程的真实状态与阻塞写入 `../../artifacts/`；本报告不预填成功结果。

## 本机执行进展

- 2026-10-23：官方源锁定安装完成，torch 2.14.1+cu130 / MuJoCo 3.16 / Warp 1.18；CUDA可用。
- 数值验证：9 passed / 0 skipped；含评测分母、尾步回归和 FK。
- 数据转换：40 clips / 2.451h，0失败。
- 集成验证首轮：17 passed，1失败。失败原因是系统 `/usr/include/python3.12/Python.h` 缺失，Triton C扩展无法编译，不是训练loss断言失败。
- 工具链修复：下载 Ubuntu `libpython3.12-dev=3.12.3-1ubuntu0.17` 到 `data/toolchain/debs/`，用 `dpkg-deb -x` 解压到项目内，`source scripts/reproduction/env.sh` 通过 CPATH 加载头文件；无需root，不改系统Python或依赖锁定。
- 后续命令前应先 `source scripts/reproduction/env.sh`。失败测试将单独重跑；已经通过的测试不反复消耗编译预算。
- 修复头文件后 WM 测试通过：8步 NSE 3.9811 → 0.2599，1步 0.4910 → 0.0711；动作输入梯度检查通过。初始27个测试现均通过（保留首轮失败日志和复测报告）。这些是小型过拟合测试的误差，不与论文模型精度直接比较。
- 冒烟训练完成：32 env、缩小网络、H=8、60仿真步、16次WM+actor+critic联合更新，checkpoint写入成功。
- 冒烟评测完成：3 clips全长口径 mean return 0.001869，weighted 0.001712，3/3失败；各clip仅存活41–46步。极短训练不应有跟踪能力，此结果用于验证早失败不再获得虚高分。
- 完整基线入口：`bash scripts/reproduction/run_lafan1_strict.sh`，固定seed0，默认4096env、完整网络、k8/H32、93维本体历史，无额外anchor输入，3小时训练预算，每2000仿真步checkpoint，结束后评测40条动作。
- 保存/恢复与编译训练测试通过（195s，含首次编译）。合计28项不同测试通过，0跳过；原工具链失败已复测通过。正式基线现在可启动。
