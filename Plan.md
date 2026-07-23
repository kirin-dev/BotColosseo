# Bot Colosseo 最终技术方案

> BotColosseo: Controllable Game Bot Styles via Skill-Preserving Policy Shaping
> BotColosseo：可控风格游戏 Bot——基于能力保持的策略塑形

## 1. 项目定位

Bot Colosseo 是一个基于 ViZDoom 的公开游戏 AI 项目，面向游戏智能体和基于游戏的 AGI 研究岗位。项目不以提出新的强化学习算法为首要目标，而以交付一个可运行、可评测、可解释、可展示的游戏 Bot 产品闭环为目标。

核心问题是：

> 如何先训练一个任务能力可靠且对多类对手具有鲁棒性的视觉 Base Bot，再在尽量保留基础能力的前提下，将它塑造成玩家能够观察和区分的多种行为风格？

项目围绕四个相互独立的产品维度组织：

- **Skill**：完成任务和对抗不同对手的能力；
- **Style**：面对相同局势时的行为偏好；
- **Difficulty**：玩家实际感受到的挑战强度；
- **Fairness**：Bot 是否遵守与玩家一致的信息边界和反应限制。

最终 GitHub 展示的主线是：

```text
Strong visual Base Bot
        ↓
skill-preserving policy shaping
        ↓
Aggressive / Defensive / Explorer
        ↓
independent difficulty controller
        ↓
automatic metrics + anonymous user evaluation
```

League、PFSP、BC、特权 Critic 等均是服务主线的技术手段，不作为 README 第一屏的中心叙事。

## 2. 已确认的范围决策

1. 最终形态是真实同步 1v1，但分阶段实现：M0/M1 使用单实例验证场景和子任务，M2 才引入双实例同步。
2. 唯一主场景为 `Crystal Run Arena`，不并行开发多个玩法。
3. Base Bot 使用 `Teacher/BC 热启动 → PPO → historical opponents/PFSP` 的训练路线。
4. 采用轻量自研 PyTorch PPO，不以 RLlib 或 Sample Factory 作为主框架。
5. 最终 Actor 使用第一人称视觉和合法公开变量；特权信息只供 Teacher、Critic、奖励与离线评测使用。
6. 三个正式风格为 `Aggressive`、`Defensive`、`Explorer`；不把容易与 Base 重叠的 Tactical 单列为首版风格。
7. 三种风格从同一个 Base checkpoint 派生为独立 checkpoint，不要求单网络实时切换风格。
8. Easy、Normal、Hard 由轻量推理控制器实现，不为每个 `Style × Difficulty` 组合重新训练。
9. 项目后期由项目作者组织约 5–10 人的匿名视频用户评测，并如实报告样本量。
10. 可用训练资源为 2 张 NVIDIA A100-PCIE-40GB；优先使用单 learner、多 CPU rollout worker，第二张 GPU 用于评测或并行风格实验。

## 3. 交付门槛

### 3.1 Resume-ready

完成以下闭环即可形成可信的简历项目：

- Strong Base Bot 通过固定能力门槛；
- Aggressive Bot 从同一 Base 派生；
- 风格指标显著改变，同时满足 Skill Retention gate；
- 完成 reward-hacking 检查、消融实验和真实对比视频；
- README 能清楚解释问题、方法、结果和限制。

### 3.2 Showcase-ready

在 Resume-ready 基础上补齐：

- Defensive 与 Explorer；
- Easy、Normal、Hard 三档难度；
- Style × Difficulty 解耦实验；
- 匿名用户评测；
- 双语 README、完整图表、演示视频、checkpoint 和一键评测入口。

项目应尽快完成 Resume-ready 垂直闭环，再扩展展示面；不能等所有风格完成后才第一次验证方法是否有效。

## 4. 游戏设计：Crystal Run Arena

### 4.1 核心规则

- 双方从对称基地出生；地图包含两条主路线和一条更长的绕行路线。
- 能量核心从中央多个候选位置中随机生成。
- 玩家接触核心后自动拾取，不设置额外 `interact` 动作。
- 携带核心返回己方得分区获得 1 分，随后核心重新生成。
- 携带者被击败后核心原地掉落；若一段时间无人拾取，核心返回中央候选点。
- 被击败者经过固定延迟后在己方出生区域重生。
- 单局目标时长约 60 秒；以得分高者获胜，平分记为 draw。
- 首版保留 ViZDoom 原生移动和 hitscan 攻击，避免自定义武器系统分散工程投入。

### 4.2 地图设计原则

- 地图规模小、结构可解释，确保训练吞吐和录像可读性。
- 三条路线在长度、暴露程度和关键区域上存在明确差异。
- 关键区域使用稳定的 region ID，供奖励和离线指标使用，但不进入 Actor 观测。
- 出生点、核心位置和少量资源位置可随机化；首版不做大型程序化地图。
- 训练、验证和测试分别使用不重叠的配置与随机种子。

### 4.3 资源许可

- 使用 Freedoom/FreeDM 资源与自制 UDMF 地图、ACS 脚本和配置文件。
- 仓库保留 Freedoom BSD 3-Clause 版权声明和来源说明。
- 不提交原版 Doom/Doom II 的商业 WAD、贴图、声音或其他资产。

## 5. 环境接口与公平边界

### 5.1 Actor 观测

Actor 的合法观测为：

\[
o_t=(I_t,v_t,a_{t-1})
\]

- `I_t`：单帧 `84×84` 灰度第一人称画面；
- `v_t`：自身生命值、弹药或攻击冷却、是否持有核心、双方公开得分、剩余时间；
- `a_{t-1}`：上一步宏动作。

CNN + GRU 负责从单帧和历史隐状态中处理部分可观测性。独立风格 checkpoint 不包含 `style_id`。

### 5.2 禁止进入 Actor 的信息

- 敌方精确坐标、朝向和墙后实时状态；
- 全局地图或 automap buffer；
- depth、labels、objects buffer；
- 不可见核心的精确坐标；
- 对手未来动作；
- 任何只为训练或评测生成的区域、路径或事件标签。

### 5.3 训练专用信息

`PrivilegedState` 可包含双方坐标、核心坐标、地图 region、可见性、交战状态和任务阶段，只允许用于：

- 规则 Teacher；
- 非对称 Critic；
- reward shaping；
- episode event 生成；
- 离线评测与 Oracle 上界。

工程上必须拆分为不同类型：

- `ActorObservation`：唯一进入 Policy 的数据；
- `PrivilegedState`：训练端专用；
- `EpisodeEvent`：拾取、掉落、得分、命中、死亡、重生、区域切换等可审计日志。

通过 observation schema 测试和模型前向接口测试防止特权字段泄漏，而非仅依赖开发者约定。

### 5.4 初始宏动作空间

首版使用 13 个离散宏动作：

```text
0  idle
1  move_forward
2  move_backward
3  strafe_left
4  strafe_right
5  turn_left
6  turn_right
7  forward_turn_left
8  forward_turn_right
9  attack
10 forward_attack
11 turn_left_attack
12 turn_right_attack
```

Normal 模式默认每个决策执行固定数量的 engine tics。动作集合只有在 smoke test 和脚本 Bot 明确证明存在表达能力缺口时才扩展。

### 5.5 两层环境

- `SingleAgentTaskEnv`：导航、拾取、返回、射击等子任务及 Teacher 验证；用于 M1。
- `SynchronousDuelEnv`：一个 rollout worker 管理 host 与 opponent 两个 ViZDoom 实例；双方均提交动作后才推进相同 tic 数；用于 M2 及以后。

上层算法只依赖统一的 `reset/step`、合法 observation 和 episode event 接口，不直接操作 ACS 或 ViZDoom 进程。

## 6. 网络结构

```text
84×84 grayscale frame
        ↓
3-layer CNN → 256-d visual feature
        +
legal scalars → small MLP
        ↓
GRU, hidden size 256
        ├── categorical policy head
        └── value path + privileged-state encoder
```

- Policy 仅接收 Actor 合法输入和 GRU hidden state。
- Critic 可以在训练时额外接收 `PrivilegedState`，形成非对称 Actor-Critic。
- 特权特征不进入 GRU 和 Policy head。
- 导出演示模型时完全移除 Critic 与 privileged encoder。
- 网络规模以训练稳定和演示帧率为准，不引入 Transformer、VLM 或大型视觉 backbone。

## 7. Strong Base Bot 训练方案

### 7.1 规则 Teacher 与脚本对手

基于地图区域图和特权状态实现有限状态 Teacher：

```text
SEARCH_CORE → PICKUP → RETURN_BASE
                    ↘ ENGAGE / EVADE
```

Teacher 负责生成稳定示范，不作为“强度上界”。脚本对手池至少包含：

- `RandomLegal`：随机但动作合法；
- `FixedRoute`：使用固定路线获取核心；
- `ObjectiveFirst`：优先完成任务；
- `AggressiveScript`：主动拦截和追击；
- `DefensiveScript`：守住关键区域和得分路线。

脚本对手的价值在于构造行为明确、可复现的能力试卷，而不是模拟高水平人类。

### 7.2 BC 热启动

- Teacher 使用特权状态产生轨迹，Actor 只从合法视觉观测预测 Teacher 动作。
- 训练集与验证/测试出生点、核心位置和种子严格隔离。
- BC 只提供导航、交互和基础交战先验，不作为最终结果。
- 保存纯 BC checkpoint，作为后续实验基线。

### 7.3 PPO 对脚本池

轻量自研 PyTorch PPO 包含：

- GAE；
- clipped policy objective；
- clipped value loss；
- entropy bonus；
- gradient clipping；
- advantage normalization；
- recurrent rollout 与 sequence minibatch；
- NaN/Inf 检查；
- optimizer、scheduler 和 checkpoint 恢复。

课程从无对手导航和完整任务开始，再逐步加入攻击、掉落、重生和不同脚本对手。密集 shaping 只用于早期学习并逐步衰减，模型选择依据独立评测而非训练 reward 峰值。

Base reward 的初始形式为：

\[
R_{base}=R_{score}+R_{win}+R_{pickup}+R_{progress}+R_{valid\_hit}-R_{death}-R_{stall}
\]

`pickup`、`progress` 和 `valid_hit` 均设置事件前置条件和单局累计上限，防止重复刷取、局部抖动和无意义攻击。

### 7.4 Historical opponents 与 PFSP

- 只有通过阶段评测的 checkpoint 才能进入历史池。
- 历史池保持约 8–12 个具有代表性的策略，避免机械堆积。
- 对手采样混合脚本、历史策略、较难策略和少量均匀采样，防止只对单一对手过拟合。
- 保存完整 cross-play payoff matrix。
- 当前策略与自身约 50% 胜率不构成强度证据。
- Base Bot 的训练在单卡上完成；第二张 A100 用于异步评测或并行实验，不建立不必要的复杂分布式 learner。

## 8. Strong Base 的定义

强度由固定评测套件定义，而不是由训练回报主观判断。

### 8.1 子任务能力

- `Goal Reach Rate`：从随机出生点到达指定区域；
- `Pickup Rate`：发现并拾取核心；
- `Return Rate`：携带核心返回己方区域；
- `Valid Hit Rate`：对可见的静止和移动目标完成有效攻击；
- `Disengage Success`：不利状态下成功脱离；
- `Full Objective Rate`：完整取得并带回核心。

### 8.2 对手池能力

- Win Rate；
- Draw Rate；
- Objective Completion Rate；
- Average Score Difference；
- Worst-case Win Rate；
- held-out 出生点、核心位置和种子表现。

### 8.3 初始能力门槛

在正式 test split 启用前冻结最终门槛。初始目标为：

- 对脚本池平均 Win Rate ≥ 70%；
- 对任一主要脚本的 Win Rate ≥ 55%；
- 无对手完整任务成功率 ≥ 90%；
- held-out 配置完整任务成功率 ≥ 80%；
- historical pool 的 worst-case 表现优于仅对固定脚本训练的 PPO；
- 使用固定评测套件运行数百局并报告置信区间。

若 pilot 表明场景固有平局率或非对称性使阈值不合理，只能在 validation 阶段基于公开规则校准一次，并记录原因；不能查看最终 test 结果后反向修改门槛。

综合能力分只用于排序：

\[
S_{base}=w_1\,WinRate+w_2\,ObjectiveRate+w_3\,NormalizedScore+w_4\,Robustness
\]

任何关键门槛未通过时，不能靠综合分中的其他高项抵消。

## 9. 风格塑形

### 9.1 总体方法

三个风格均从同一 Strong Base checkpoint 初始化，并在相同任务和对手分布下训练：

\[
J_{style}=J_{PPO}(R_{task}+\lambda_sR_{style})
-\beta_{KL}D_{KL}(\pi_s(\cdot|o)\Vert\pi_{base}(\cdot|o))
\]

能力保持采用三层保护：

1. 冻结视觉 CNN；
2. 首轮只训练 residual style adapter 与 policy head；若风格不足，再以更低学习率解冻 GRU；
3. checkpoint 必须通过硬性 Skill Retention gate。

风格选择扫描一个有界的 `lambda_style × beta_KL` 小网格，报告 Style Fidelity 与 Skill Retention 的 Pareto 前沿，选择折中点而非单一最高风格奖励。

### 9.2 Aggressive

目标行为：更主动拦截、发起有效交战并进行有限时间追击，同时仍能完成核心运送。

主要事件与指标：

- Engagement Initiation Rate；
- Valid Attack Rate；
- Chase Duration；
- Engagement Distance；
- Retreat Rate；
- Objective Progress Lost to Chase。

只在对手可见、距离合理或近期存在可靠视野记忆时奖励交战。对墙射击、无目标射击和无任务进展的超长追击必须受到惩罚或不计奖励。

### 9.3 Defensive

目标行为：优先保护核心、己方得分区和关键路线，在不利交战中更早脱离，但仍具备主动得分能力。

主要事件与指标：

- Defensive Zone Occupancy；
- Opponent Denial Rate；
- Core Recovery Rate；
- Low-health Disengage Success；
- Survival Time；
- Own Objective Completion Rate。

只有防守行为与核心、对手或得分风险相关时才给予奖励；与局势无关的原地驻守不获得收益。

### 9.4 Explorer

目标行为：使用更多路线和绕行选择，降低路径重复，同时维持任务效率。

主要事件与指标：

- normalized Route Entropy；
- Unique Region Visits；
- Map Coverage；
- Path Repetition Rate；
- Flank Usage；
- Objective Efficiency。

奖励新的有效区域转移和完成任务时的路线多样性；不直接奖励频繁转向，避免原地旋转或无意义绕路。

### 9.5 Reward-hacking 约束

每个风格奖励项必须具有：

- 明确事件定义；
- 生效前置条件；
- 单局累计上限；
- 与之对应的自动指标；
- 至少一个反例单元测试；
- 分项日志和轨迹回放检查。

## 10. Skill Retention 与风格成功标准

风格模型相对 Base 的能力保持定义为：

\[
SkillRetention(s)=\frac{Performance(\pi_s)}{Performance(\pi_{base})}
\]

初始硬门槛：

- 综合 Skill Retention ≥ 85%；
- 对任一主要脚本对手的相对表现 ≥ Base 的 75%；
- 主要目标风格指标相对 Base 方向正确且在配对评测中稳定；
- 与另外两种风格的指标向量具有可观测差异；
- 无明显 reward hacking；
- Demo 行为来自正式 checkpoint 和真实 episode 日志。

若 Base 在某个分母指标上接近零，则不使用比率，而报告绝对差值和置信区间。

## 11. Style 与 Difficulty 解耦

Difficulty 不重新训练策略，也不改变生命值、伤害或信息边界。推理控制器只能调整：

- policy update/decision interval；
- observation-to-action reaction delay；
- 有界转向或瞄准噪声；
- action temperature；
- pursuit interruption window。

Normal 使用原始策略参数；Easy 增加反应延迟和有界误差；Hard 减少这些限制，但仍遵守最小反应时间、最大转向速度和公平观测边界。

验收要求：

- 同一风格下，Easy → Normal → Hard 的 Win Rate 或挑战性近似单调；
- 同一难度下，三种风格仍可区分；
- 任何难度都保持基本导航和任务能力；
- Easy 不是随机动作，Hard 不是 Oracle。

具体参数通过 validation 对局校准后写入 `configs/difficulty.yaml`，test 阶段冻结。

## 12. 实验协议

### 12.1 数据与配置隔离

- `train`：允许课程学习、随机化和对手采样；
- `validation`：用于 checkpoint 选择、奖励权重和难度参数调节；
- `test`：冻结后不参与任何调参，只用于最终报告。

模型比较使用相同地图配置、对手、随机种子和初始条件做配对评测。核心结果报告局数、均值及置信区间。

### 12.2 必要实验

1. **Base 能力**：BC、PPO 对脚本、PPO + historical/PFSP。
2. **风格对比**：Base、Aggressive、Defensive、Explorer。
3. **能力保持消融**：reward only、reward + KL、reward + KL + freeze/adapter。
4. **Style × Difficulty**：验证风格稳定性和难度单调性。
5. **Fairness**：最终 Fair Actor 与仅作上界的 privileged Oracle。

### 12.3 评测矩阵

| 维度 | 主要证据 |
|---|---|
| Skill | Win Rate、Objective Completion、Score、Worst-case、held-out 表现 |
| Style | 三种风格目标指标相对 Base 的变化 |
| Retention | Style Bot / Base Bot 的归一化能力比与绝对差值 |
| Diversity | 区域访问、路线转移和动作分布的两两差异 |
| Difficulty | 同风格下挑战性随 Easy → Hard 的变化 |
| Fairness | Actor 输入审计、墙后追踪测试、反应与转向限制 |
| User Experience | 匿名风格识别、公平性、自然度、趣味性评分 |

## 13. 用户评测

- 邀请约 5–10 名测试者观看匿名、随机顺序的正式 checkpoint 对局视频。
- 每名测试者先猜测风格，再以 5 分 Likert Scale 评价挑战性、公平性、自然度和趣味性。
- 可增加匿名配对问题，例如“哪个更激进”“哪个更像真人”“哪个更适合新手”。
- 保存匿名原始响应和生成统计图的脚本。
- 如实报告样本量、缺失响应和不确定性；小样本结果只作为产品感知验证，不声称具有大样本统计显著性。

## 14. 测试与工程质量

### 14.1 单元测试

- 各奖励项的正例、反例、上限和 reset；
- 风格指标与区域转移统计；
- difficulty 参数和动作合法性；
- Actor observation schema 不包含特权字段；
- GAE、advantage normalization 和 PPO loss 数值；
- RNN hidden state 在 episode、death/respawn 边界的处理；
- terminal 与 truncation 区分；
- checkpoint 保存和恢复；
- NaN、Inf 和动作越界检查。

### 14.2 场景与集成测试

- 拾取、得分、掉落、核心返回；
- 死亡、重生和 episode timeout；
- 双方 tic 一致和同步结束；
- 异常 ViZDoom 进程能够清理；
- headless、render 和录像模式互不干扰；
- 固定短轨迹的事件日志和指标回归。

任何里程碑只有在对应 smoke/integration gate 实际通过后才能进入下一阶段。`PROCESSING COMPLETE`、单次成功录像或训练 reward 上升均不能替代验收证据。

### 14.3 可复现性

每次正式实验保存：

- 模型、优化器、scheduler 和训练步数；
- 完整解析后的配置；
- Git commit、环境版本和随机种子；
- 对手池版本与 payoff matrix；
- 分项训练奖励；
- validation/test 评测结果；
- CSV/JSON 原始日志和图表生成脚本。

## 15. 仓库结构

```text
BotColosseo/
├── assets/
│   └── scenarios/             # UDMF/WAD/ACS/CFG 与资产许可
├── src/botcolosseo/
│   ├── envs/                  # 子任务、同步 1v1、观测与事件
│   ├── agents/                # CNN-GRU、BC、adapter、checkpoint
│   ├── training/              # PPO、rollout、对手池、PFSP
│   ├── shaping/               # 风格奖励与 difficulty controller
│   ├── evaluation/            # skill/style/fairness/user study
│   └── demo/                  # 录像、overlay 与策略选择
├── configs/
├── scripts/
├── tests/
├── docs/
├── README.md
├── README_CN.md
├── LICENSE
└── pyproject.toml
```

保持模块边界清晰，但不为架构美观提前拆出无实际用途的类和目录。

## 16. 里程碑与阶段门禁

### M0：环境可靠性

交付：

- 正确的 Python 3.10 环境入口；
- ViZDoom headless init、frame、action、termination、reset；
- 固定种子；
- render 与录像；
- 进程和音频错误处理。

门禁：所有 smoke tests 在目标运行环境实际通过。

### M1：单实例任务原型

交付：

- Crystal Run 地图、ACS 与 cfg；
- 导航、拾取、返回、射击子任务；
- region graph 和 EpisodeEvent；
- 至少五类规则 Teacher/脚本策略；
- 场景与奖励反例测试。

门禁：脚本 Teacher 能稳定完成对应子任务，事件日志与规则一致。正式量化标准为
navigation/pickup/return ≥95%、static hit ≥90%、moving hit ≥75%，每项使用
100 个冻结 test seeds，且事件协议不一致数必须为 0。

状态：已于 2026-07-20 通过。五项均为 100/100，协议不一致数为 0；原始证据位于
`reports/m1/`，公开展示位于 `docs/assets/`。

### M2：真实 1v1 与 Base 训练

交付：

- `SynchronousDuelEnv`；
- demonstration dataset；
- BC checkpoint；
- recurrent PPO；
- 脚本对手训练与固定评测。

门禁：双实例长时间同步稳定，PPO 明确优于纯 BC 和随机策略。

状态：2026-07-21 官方 1,500 局评测工程门通过（完整性、配对、同步、协议和产物
不一致数均为 0），但能力门未通过。PPO 总胜率 77.0%，BC 为 75.2%，随机策略为
34.4%；PPO 未达到相对 BC +10pp 的冻结门槛，objective completion 为 93.2%
（BC 97.8%），对 `objective_first` 胜率为 23%。该结果如实保留为 M2 FAIL。
经评审，M3 只允许显式使用 integrity-qualified 路线继续：不得使用 M2 test rows
调参，需另跑 validation-only anchor evidence，且最终必须独立通过 M3 全部门禁。

### M3：Strong Base

交付：

- historical policy pool；
- PFSP；
- cross-play payoff matrix；
- held-out 与 worst-case 评测。

门禁：通过第 8 节冻结的 Strong Base 能力门槛。

状态：2026-07-22 已完成 historical pool、PFSP、cross-play、候选选择和证据审计，
工程与协议完整性合格，但真实结果未通过全部冻结能力门槛。因此选定的 200k checkpoint
只作为 `integrity-qualified capability anchor`，不声称 M3 正式通过。后续风格实验使用该
anchor 是经显式评审批准的产品路线继续，不会反向把 M4 的成功包装成 M3 PASS。

### M4：Aggressive 垂直闭环

交付：

- style adapter、KL 与 reward；
- reward-hacking tests；
- lambda/KL 消融；
- Style–Skill Pareto 图；
- Base/Aggressive 对比视频。

门禁：Aggressive 通过风格与 Skill Retention 标准。此时达到 Resume-ready。

状态：已于 2026-07-23 通过。固定 alpha 0.25 checkpoint 完成 200 局 validation
配对评测，全部七项门禁通过：Strong Base/Aggressive 胜率为 87.0%/89.0%，Skill
Retention 为 1.000，engagement shift 为 +0.0998/100 decisions，其 95% bootstrap
区间为 `[0.0460, 0.1706]`。正式 GIF、MP4、指标卡和 hash-bound publication
manifest 已发布；它们明确是 validation 展示，不冒充 test 结果。此时项目达到
Resume-ready。

### M5：完整风格产品

交付：

- Defensive 与 Explorer；
- 三档 difficulty controller；
- Style × Difficulty 评测；
- 三风格匿名演示素材。

门禁：三种风格均满足能力保持，难度基本单调且不破坏风格辨识度。

当前进展：Defensive 已完成风险条件 Teacher、success-filtered data、adapter
distillation、固定 alpha grid、配对评测和证据审计的工程实现，但现有候选尚未
稳定通过 protective-presence gate。Explorer 已完成 score-conditioned 三路线
Teacher、成功窗口平衡采样、adapter distillation、固定 alpha grid、配对评测、
确定性选择和 hash-bound 证据审计的工程实现；生产实验结果尚未冻结。因此暂不
声称 Defensive、Explorer 或 M5 通过。difficulty controller、四策略展示和用户
评测仍待完成。

### M6：公开发布

交付：

- 用户评测及真实数据图表；
- 15–30 秒三风格 GIF；
- 各风格正式视频；
- 双语 README；
- checkpoint、配置和一键评测/Demo 命令；
- 资产与代码许可说明。

门禁：README 中所有完成声明均有日志、测试、模型或可运行 artifact 支撑。此时达到 Showcase-ready。

## 17. GitHub 展示结构

README 第一屏按以下顺序组织：

1. 三风格同场景对比 GIF；
2. 一句话问题定义；
3. Skill–Style–Difficulty–Fairness 四维解耦图；
4. Base 与三个风格的真实指标表；
5. 一键 Demo 与评测命令。

后续依次说明：玩法、训练路线、风格塑形、实验、用户评测、公平约束、复现、资产许可和已知限制。图表必须由提交的原始日志和脚本生成，不手工伪造。

计划生成的核心展示物：

- `base_bot_demo.mp4`；
- `aggressive_bot_demo.mp4`；
- `defensive_bot_demo.mp4`；
- `explorer_bot_demo.mp4`；
- `style_comparison.gif`；
- `style_metrics.png`；
- `skill_style_pareto.png`；
- `route_heatmaps.png`；
- `difficulty_comparison.png`。

## 18. 明确排除范围

首版不实现：

- 长期记忆、LLM planner 或 VLM 微调；
- 世界模型或 Diffusion Policy；
- Tactical 独立风格；
- 统一 style-conditioned policy；
- 大规模 League Exploiter；
- 3v3 团队协作；
- UE5、Unity 或复杂 Web 后端；
- 大规模分布式 learner；
- 大型程序化地图；
- 通过增加生命值、伤害或隐藏信息制造难度。

## 19. 已知风险与应对

### ViZDoom 多人同步复杂

先完成单实例任务和事件协议，再引入双实例；将每局对局封装在独立 worker 中，并设置初始化、step 和退出超时。

### 视觉 PPO 早期学习慢

使用特权 Teacher 示范和 BC 热启动；Critic 可使用特权状态，但 Policy 严格保持公平观测。

### Base 胜率缺少标准基准

预先冻结脚本能力试卷、historical cross-play、held-out 配置和子任务门槛，不以自博弈 50% 或训练回报替代强度证据。

### 风格奖励导致能力退化或投机

采用 adapter/freeze、KL、任务奖励保留、Skill Retention hard gate、奖励上限和反例测试；从 Aggressive 单风格垂直验证后再扩展。

### Explorer 退化为乱走，Defensive 退化为驻守

所有风格奖励与有效任务机会绑定，同时监控 Objective Efficiency 与完成率。

### 用户样本量较小

将用户评测定位为轻量产品感知验证，公开原始匿名响应与样本量，不夸大统计结论。

## 20. 当前仓库与环境事实

截至 2026-07-20：

- 仓库已初始化并采用小步、可审查提交记录。
- 已存在 `botcolosseo` Conda 环境，Python 3.10.20、PyTorch 2.6.0+cu124、ViZDoom 1.3.0。
- 当前 shell 的 `python` 错误指向另一个 Python 3.7 环境，正式脚本不能依赖隐式 PATH。
- M0 的真实 ViZDoom、termination/reset、确定性动作和 MP4 门禁已通过。
- M1 的场景、ACS 事件协议、单实例环境、五类 Teacher 与 500 回合冻结评测已完成。
- M1 正式结果为五项 100/100、事件协议不一致数 0；这不是 learned-agent 或 multiplayer 结果。
- M2 工程与 1,500 局正式评测已完成，但冻结能力门失败；该失败结果被保留。
- M3 工程与完整性证据已完成，但并非正式 M3 能力门 PASS；其选定模型只作为后续
  风格塑形的 capability anchor。
- M4 Aggressive 已完成并通过冻结 validation gate，仓库已经 Resume-ready。
- 2 张 NVIDIA A100-PCIE-40GB 已在真实训练与评测会话中验证可用。

## 21. 紧接着的实施顺序

1. ~~修复环境入口并完成 M0 gate。~~
2. ~~初始化 Python package、Git 仓库和测试框架。~~
3. ~~制作 Crystal Run 地图、ACS 事件协议与单实例环境。~~
4. ~~完成五类 Teacher 和冻结 train/validation/test 配置。~~
5. ~~通过 M1 的 500 回合正式 gate 并发布证据。~~
6. ~~实现 M2 `SynchronousDuelEnv`、demonstration dataset、BC/PPO 和正式评测；
   如实保留能力门失败。~~
7. ~~实现 M3 historical/PFSP 与证据审计；将未完全通过门禁的模型明确标为
   capability anchor。~~
8. ~~完成 M4 Aggressive 风格、200 局配对门禁和公开展示，达到 Resume-ready。~~
9. 完成 M5 Defensive、Explorer、difficulty controller 与 Style × Difficulty 评测。
10. 完成 M6 用户评测、四策略展示、双语文档、checkpoint 与一键复现入口。

后续实现必须按里程碑逐个验证；如果某阶段的真实证据与本方案假设冲突，应更新本文件中的具体参数和风险判断，但不得悄然改变项目主线、评测隔离或公平边界。
