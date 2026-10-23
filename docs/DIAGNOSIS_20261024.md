# 2026-10-24：失败定位、修复与受控改进

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent + karpathy-guidelines
- Origin Mode: run / reproducibility diagnosis
- Origin Date: 2026-10-24
- Verification Status: 代码修复已验证；训练改进效果待对照评测

## 已验证的问题

| 项目 | 证据 | 处理 |
|---|---|---|
| replay终止标签错位 | row t的done表示a_t导致的终止，sample_windows却读取row t+1；两条回归测试均在修复前失败 | 改为读取与动作/奖励相同的row t |
| checkpoint续训缺少replay预热 | 保存的iteration很大，但恢复后buffer为空；原测试只构造Trainer，没实际继续run。新增run在修复前触发no sampleable rows | 按本次实际采集步数判断warmup，已测试保存→恢复→继续更新 |
| LAFAN全局漂移不可观测 | 固定参考，机器人平移(1,.5)m、yaw旋转.8rad，策略输入差6.7e-8、动作差5.1e-7；root位置奖励从.00998降到8.5e-9 | 将相对anchor位姿作为明确标记的增强goal做对照实验，原93维本体观测不变 |

完整测试：34 passed / 0 skipped。新增warm-start对9维goal扩展的首层权重插入全零列，critic也在goal对应位置插零；测试确认新增特征取任意值时，初始化actor/critic输出仍与原checkpoint相同。两种消融均重置优化器/计数器、保留相同模型权重与采样器状态。

## 已排查但不能当作主要原因的项目

参考初始化的soft-limit裁剪影响15,392条BONES动作首帧；其中160条曾在1秒内失败。不过固定最终checkpoint、同一主机/256env/705条失败子集的配对评测：

| reset限制 | mean return | 失败 |
|---|---:|---:|
| soft（原设置） | .22728 | 686/705 |
| hard（保留合法参考角） | .23035 | 681/705 |

12条恢复、7条退化，净减少5次失败，不能解释大部分差距。原129,943全库评测为705次失败，与这里686不同；这里运行主机/并行batch/子集排队不同，必须比较同组soft/hard，不能把705→681宣传为模型训练改善。默认仍为soft，hard保留为显式诊断参数；评测JSON现在记录protocol。

## 接下来的受控实验

从同一LAFAN1三小时checkpoint出发，seed0，4096env、H32/k8、默认网络与物理参数，分别：

1. `lafan_goal_control_20261024`：原680维goal。
2. `lafan_goal_anchor_20261024`：689维goal，新增参考相对位姿9维。

两组都修正终止标签；相同新采样5000个向量步、约1984次联合更新，而非仅比较不同墙钟预算。LR保持320,000总步/250,000衰减的原时钟；新增`stop_at_iteration`只截短实验，不压缩LR曲线。独立run、独立checkpoint，结束后全40条整段评测。单组安全时限1h。

增强goal改变了输入信息，属于对当前实现选择的改进消融；论文未公开完整goal特征定义，不能据此直接声称论文采用或禁止该输入。相对global pose在真实部署中需要相应状态估计。

BONES后续将从原checkpoint续训，先保持原goal和物理配置，仅修正已证实的训练问题，固定2000条子集及全库评测；对Sports/Stunts进一步检查初始接触和模型误差。两轮原始基线结果保留。

## 追加诊断

原LAFAN checkpoint在相同40条动作首帧起步、每条只评测前10秒时：mean return **0.887517**、0/40失败；整clip则0.592232、1/40失败。此处是评测时限对照，不是新模型改善，也不是已确认的作者LAFAN评测协议。它与根部不可观测实验一起支持长期漂移是重要差距来源。结果位于 `../../artifacts/lafan_prefix10s_diagnostic.json`。

初始接触静态CPU检查（705失败+705随机成功，完全相同场景模型）中：失败组13条初始ncon>35、1条nefc>250，成功组分别1条/0条。**nconmax是共享接触池的容量估计，单环境ncon>35不代表溢出**；njmax才是逐环境约束上限。该检查不能证明轨迹运行中发生过溢出，因此未据此改变物理容量或求解器。

受控goal实验结束后，`.213`排队执行原BONES最终checkpoint续训2小时，保留原goal和soft reset，修复终止标签和空buffer预热。最终仍评测同一2000条子集及全库。该续训结果包含额外训练计算，不能把全部收益归因于标签修复。
