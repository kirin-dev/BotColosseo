# BotColosseo

**面向搜打撤的可控风格 Game Bot**

[English](README.md)

在一个紧凑型 1v1 搜打撤任务中，先训练具备任务能力的第一视角 ViZDoom Bot，
再派生三种可辨识的学习型风格。

## [打开可交互展示页 →](https://kirin-dev.github.io/BotColosseo/)

展示页包含四段可直接播放的视频、场景平面图、训练方法、风格判别方式和最相关
的量化结果。

## 任务是什么？

```text
搜索物资 → 交战或脱离 → 管理背包 → 撤离 → 带出价值
```

- 单局 75 秒，双方各 100 HP，每次有效命中造成 20 点伤害。
- 初始 30 发子弹，不换弹、不复活；场内有两个中立撤离点。
- 三格背包装载价值为 10、25、50 的物资，高价值物资会替换最低价值物资。
- 死亡会将全部未带出物资变成可被对手拾取的尸体缓存。
- 击杀本身不计分，只有成功撤离带出的价值有效。

## 一个 Base，四种可见行为

| Bot | 行为侧重 | 代表性因果链 |
|---|---|---|
| **Strong** | 均衡完成任务 | 搜索 → 高价值物资 → 撤离 → 带出 |
| **Aggressive** | 将有效交战转化为收益 | 命中 → 击杀 → 尸体缓存 → 撤离 |
| **Defensive** | 风险下保护携带价值 | 停止追击 → 脱离 → 保值撤离 |
| **Explorer** | 有效路线和物资多样性 | 多区域搜索 → 背包升级 → 撤离 |

Strong CNN-GRU Actor 依次经过脚本 Teacher 数据、行为克隆、循环 PPO、历史对手
和轻量 PFSP 训练。三个风格是绑定同一个冻结 Strong Actor 哈希的有界残差
logit adapter。风格奖励由仅训练期使用的机会检测器按状态激活；部署策略仍然
只使用相同的公开观测。

收尾审查在冻结 Strong checkpoint 训练完成后修复了 PFSP 平局记账。下方闭环
评测结果不受影响，但本版本不声称 PFSP 带来了确定因果增益；重训练留待后续。

### 当前 v3 代码入口

| 层级 | 入口 |
|---|---|
| 游戏规则与 ACS 地图 | `assets/scenarios/crystal_run_extraction_randomized/` |
| 同步环境与公开协议 | `src/botcolosseo/envs/synchronous_extraction.py` |
| CNN-GRU Actor 与非对称 Critic | `src/botcolosseo/agents/extraction_model.py` |
| BC、循环 PPO、PFSP 与风格塑形 | `src/botcolosseo/training/extraction_*.py` |
| 冻结评测与证据分层 | `src/botcolosseo/evaluation/extraction_*.py` |
| 完整运行命令 | `script.md` |

## 核心结果

| Strong 能力 | 结果 |
|---|---:|
| 随机布局 Validation 撤离率 | **83.3%** |
| 随机布局 Validation 胜率 | **56.7%** |
| 随机布局 Validation 平均带出价值 | **39.10** |
| 随机布局 Heldout 撤离率 | **85.8%** |

### 随机布局发布版本

当前 Strong 与三个风格 adapter 共享同一个随机物资场景和冻结 Strong
checkpoint。7 件物资在 16 个安全点位间按无碰撞排列生成；这是有限布局族上的
domain randomization，不代表连续坐标泛化。20 个 checkpoint 先各做 32 局冻结
筛选，再对选中的 950k checkpoint 做独立 240 局确认。详见
[派生曲线数据](reports/extraction/training-curve.json)。

| Bot | 公开证据层 | 选中视频实际证明的行为 |
|---|---|---|
| Aggressive | 代表性案例 | 5 次命中 → 击杀 → 尸体缓存 → 带出 85 价值 |
| Defensive | 代表性案例 | 携带价值时脱战 → 撤离，0 击杀 |
| Explorer | 代表性案例 | 搜索 4 个物资区域 → 背包升级 → 带出 85 价值 |

这些是从 validation 选择的产品演示，不代表所有风格在完整分布上均有提升。
每段视频均明确限定为代表性案例，并包含由引擎事件记录的完整行为因果链。
完整案例、checkpoint/视频哈希、证据层级与研究检查见
[机器可读审计](reports/extraction/showcase/audit.json)。

## 证据边界

部署时 Actor 只能接收 84×84 第一视角灰度图及自身公开状态，不能接收敌方
血量或位置、世界坐标、深度图、目标标签、俯视图或观众遥测。特权状态仅用于
非对称训练 Critic、训练 reward shaping，以及离线评测与观众遥测，不会进入
部署 Actor。

当前公开结论限定为产品 Showcase，不声称全风格 benchmark 成功、聚合能力保持
成功或 official-test 结果。
候选选择阶段禁止访问 test。冻结协议规定每个策略仅测试一次 400 局，
official test 总计 1,600 局；当前尚未运行。

## 复现

使用 `botcolosseo` Conda 环境。完整命令见 [script.md](script.md)，冻结门见
[Plan.md](Plan.md)。

```bash
conda activate botcolosseo
python -m pip check
python -m ruff check src tests scripts
python -m pytest tests/unit -q
```

项目源码采用 MIT License。ViZDoom 与 Freedoom 保留各自许可证，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
