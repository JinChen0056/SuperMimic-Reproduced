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

## BONES-SEED 数据准备与首轮真实评测

官方来源：https://huggingface.co/datasets/bones-studio/seed ，已查询并记录 revision `2f59b2077b9da34dd4e43618e705c7cb962c9a66`。本机已有原始归档，不重复下载或修改：

- G1压缩包：`/home/mai-4090-02/chenjin/mainchain/data/bones_seed/source/g1.tar.gz`，23,499,973,647 bytes；实际 SHA256 与官方 LFS SHA256 完全一致：`52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a`。
- metadata v004：142,220 条，288.3208h @120Hz；实际 SHA256 与官方一致：`7cbe5624f91ae19af3feee69cff74b2cf553fb26367c8edc4b674c3d8ea97e7d`。
- 新入口 `scripts/reproduction/prepare_bones_stream.py`：逐条读取归档，用原转换器处理厘米/欧拉角度/关节角度与120→50Hz重采样、FK和可行性筛选，再直接追加到fp16紧凑状态库。仅临时保留每worker一份CSV/NPZ，不全量解压，归档只读。保留逐条筛选manifest；磁盘保留8GiB，最终atomic打包时检查额外空间。
- 前128条端到端试运行：117条保留，数据有限且名字唯一；第一条保留动作的紧凑状态与原CSV→NPZ→pack流程逐元素相同。随后已启动全量处理，CPU nice=15、4workers，与现有GPU训练并行。
- **筛选尚未严格对齐**：现有代码名称筛选后130,137条，论文129,667条，差470；完整筛选后统计另行记录。源码未公开精确的保留动作清单，不能为凑条数随机删除动作。当前产物命名为近似筛选版本，不据此声称完全复现主结果。

LAFAN1首个正式checkpoint（iteration2000，784次联合更新，约11分钟，含初始编译）：40条完整动作评测 mean return **0.0067**、length-weighted **0.0062**、**40/40失败**。这是早期结果，尚未复现论文效果。评测与训练短暂同卡并行，本次运行不用于严格的GPU效率比较。训练episodic return与模型误差是诊断量，不能替代完整clip评测。

数据条数与耗时不能按比例换算：LAFAN1的40条共2.45h，平均每条约220s；BONES-SEED的142,220条共288.32h，平均每条约7.3s。在线训练使用固定env数、固定batch的重复交互与更新，并非一次遍历静态数据的监督训练。

评测协议的未决项：论文 Appendix F 明确描述的是 BONES-SEED benchmark；LAFAN1 的 Fig.4 只明确写了对40条clip打分，未给出是否整条播放或评测episode时限。本分支当前将 Appendix F 的首帧整条播放规则用于 LAFAN1，属于明确记录的实现选择，不能称为已逐项对齐作者的 LAFAN1 evaluator。后续比较必须同时说明这一点。

全量准备结果（本机实跑）：142,220条全部处理；名称过滤130,137条，几何可行性再去除194条（elevated132 / penetration46 / floating16），最终 **129,943条 / 263.7538h / 47,475,680帧@50Hz**。相对论文129,514条 / 264.56h，多429条、少0.8062h；不能称为精确筛选复现。紧凑库6.74GB（6.3GiB），处理与打包611.7秒，原归档保持只读。完整manifest覆盖142,220条；CPU训练加载器已验证clip/frame数量，抽取首/中/尾状态生成的10帧goal为有限的 `(3,680)`。最终统计与校验在 `data/library/bones_seed_compact.summary.json`，尚未启动BONES-SEED训练。

LAFAN1第二个checkpoint（iteration4000，1584次联合更新，约20分钟训练，含编译）：40条整段评测 mean return **0.2741**、length-weighted **0.2850**、**29/40失败**，11条完成整段；平均执行时长由首checkpoint的2.64秒增加到161.54秒。该指标明显改善，但未达论文0.82目标；原3小时基线继续运行。第二次评测仍与训练同卡重叠，故墙钟速度仅作运行诊断。

## 第二台机器的 BONES-SEED 并行训练

新增 `scripts/reproduction/run_bones_strict.sh`：默认 seed0、4096 env、policy batch4096、完整网络、k8/H32、论文93维本体观测，无额外anchor。训练预算3小时，结束后先评测固定seed0的2000条子集，再评测全库。`scripts/eval_benchmark.py --library <packed.pt>` 现在直接支持紧凑库，并按库内clip数量设置并行环境；新增回归测试验证多clip库不会被当成单clip。`build_bones_eval_subset.py` 固定抽样索引与名字manifest。本机新增路径验证：5项benchmark测试通过，8条真实BONES动作的packed-library评测完整跑通。

目标`.213`配置：RTX4090 / 48GiB、Ubuntu22.04、Python3.12.13，依赖仍为原uv.lock（torch2.14.1+cu130 / Warp1.18 / mjlab固定commit）。代码由Git bundle同步，6.74GB紧凑库经LAN复制后SHA256与本机一致。复用本机uv缓存，离线安装完成；远端11项预检测试通过，CUDA、torch.compile和反向传播通过。

该机已加载内核驱动595.91.07，系统用户态库却是595.99.02。为避免重启或修改系统，将NVIDIA官方595.91.07 redistributable解压至项目 `data/toolchain/nvidia-595.91.07`，通过 `nvidia-active` 链接和仅本进程的 `LD_LIBRARY_PATH` 使用。注意同时提供 `libcuda.so` 和 `libcuda.so.1`：Warp优先dlopen前者，而PyTorch使用后者，两者必须指向同一匹配版本。该目录不提交Git；来源为 `https://developer.download.nvidia.com/compute/nvidia-driver/redist/nvidia_driver/linux-x86_64/nvidia_driver-linux-x86_64-595.91.07-archive.tar.xz`。系统库和内核未改动。

## LAFAN1 三小时结果与 BONES-SEED 中期结果

LAFAN1已于2026-10-23 19:50（本地时间）完成训练及评测，退出码0。实际训练43,750个向量仿真步、17,484次联合更新，预算3小时（非320,000步完整训练）。最终40条整段mean return **0.592232**、length-weighted **0.590156**，失败 **1/40**；失败动作 `fight1_subject2` 在5457/12243个参考transition处终止。尚未达到论文Fig.4的0.82目标，且LAFAN1评测时限仍有前述未决项。

奖励诊断：相对身体位置奖励约为该项上限96%，身体朝向约80%，线/角速度各约58–59%；全局root位置约5.5%、root朝向约21.6%。因此相对身体姿态已较好，但全局根部漂移及速度误差仍显著。优先检查goal坐标表示/相对根部信息和速度跟踪；这是依据分项奖励提出的诊断方向，不是已证明的失败原因。论文未给出足够完整的goal特征定义，当前680维表征属于本复现的实现选择。

BONES-SEED第34,000步checkpoint（约2.5小时）在固定seed0的2000条子集上：mean return **0.872287**、length-weighted **0.8657**，失败 **17/2000**，1983条完成整段。为不干扰远端训练，将该checkpoint和相同子集拷到本机空闲GPU离线评测；两机均为RTX4090，软件锁定相同，驱动版本不同，故该结果标记为本机离线诊断。结果存于 `logs/bones_remote_eval/benchmark_subset2000_ckpt_0034000_local.json`。远端3小时训练尚未结束，结束后原计划会在远端评测固定子集及全库。

LAFAN1与BONES-SEED的动作长度/难度分布不同，不能用BONES子集0.8723超过LAFAN1目标0.82宣称复现成功。BONES主结果仍需对照论文全量回报、失败分组及精确筛选范围；当前筛选差异仍为多429条、少0.81h。

## 首轮双机训练最终结果（2026-10-24核验）

两项训练与各自最终评测均已正常结束，退出码0，当前没有这两项训练进程。

| 运行 | 训练预算 | 向量仿真步 / 联合更新 | 评测范围 | Mean / length-weighted return | 失败 |
|---|---|---|---|---|---|
| 本机LAFAN1 | 3h | 43,750 / 17,484 | 全40条 | 0.592232 / 0.590156 | 1/40 |
| `.213` BONES-SEED | 3h | 41,150 / 16,444 | 固定2000条子集 | 0.888807 / 0.880714 | 12/2000 |
| 同一BONES最终checkpoint | 同上 | 同上 | 全129,943条 | **0.888727 / 0.882975** | **705/129,943** |

LAFAN1训练及评测于2026-10-23 19:50完成；BONES训练约20:58结束，子集和全库评测于21:15完成。远端原始结果、配置、provenance和stdout已镜像到本机 `logs/bones_remote_final/`，checkpoint仍保存在两边对应run目录；原始模型和逐动作结果不提交Git。

**结论：方法主流程及大规模训练评测已跑通，但首轮尚未完整复现论文成绩。** LAFAN1回报仍明显低于0.82；BONES平均回报接近论文3h单H100的>0.90，但705次失败仍明显多于论文25/129,514，且硬件、筛选集合存在差异。完成率99.457%不能替代上述回报与失败数对比。两轮都只是3小时预算试跑，未跑满配置中的320,000向量步；iteration时钟与论文优化更新次数仍需进一步对齐。

BONES失败诊断（将全库结果按源metadata分类）：Sports 305/3831失败，Stunts 164/287失败，两类合计469/705次失败；Basic Locomotion Neutral为48/28440。名称含handstand的失败160条；总失败中194条在1秒内终止，失败动作的中位执行时长1.86秒。优先排查初始参考状态/接触、倒立与翻滚等稀有姿态的模型误差和训练覆盖。以上是统计定位，不代表已确认失败原因。分组CSV与汇总位于 `../../artifacts/bones_final_failure_categories.csv`、`../../artifacts/bones_final_summary.json`。
