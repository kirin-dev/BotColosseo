# Crystal Run: Extraction

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
logit adapter；公开推理不包含规则式风格 governor。

## 核心结果

| Strong 能力 | 结果 |
|---|---:|
| Solo 撤离率 | **100%** |
| 脚本对手胜率 | **89.2%** |
| Validation 撤离率 | **94.6%** |
| Heldout-layout 撤离率 | **85.8%** |

| 风格 | 成对 validation 风格偏移 | 成对任务保持率 |
|---|---:|---:|
| Aggressive | **+0.101** | **93.5%** |
| Defensive | **+0.006** | **96.3%** |
| Explorer | **+0.050** | **94.4%** |

三个风格使用各自的风格指标，偏移数值不能用于跨风格强弱排名。完整案例、模型
哈希、证据层级及失败项见
[机器可读审计](reports/extraction/showcase/audit.json)。

### 匹配 200k 风格消融

每格为“成对风格偏移 / 成对任务保持率”。

| 变体 | Aggressive | Defensive | Explorer |
|---|---:|---:|---:|
| Full | +0.023 / 91.1% | +0.006 / 96.3% | +0.004 / 91.6% |
| Reward + KL | +0.017 / 92.1% | -0.031 / 92.1% | -0.031 / 87.9% |
| Reward only | +0.082 / 94.4% | +0.081 / 88.8% | -0.016 / 87.4% |

九格均使用同一个冻结 Strong Actor、训练预算、场景、协议和 240-case validation
划分。结果体现风格相关的取舍：移除正则可增大 Aggressive 或 Defensive 的测量
偏移，但 Explorer 会反向并触发反作弊失败。各列风格指标不可横向比较。完整
哈希、门禁和失败项见[消融审计](reports/extraction/style-ablation.json)；
`test_cases_accessed=false`，未打开 official-test cases。

## 证据边界

部署时 Actor 只能接收 84×84 第一视角灰度图及自身公开状态，不能接收敌方
血量或位置、世界坐标、深度图、目标标签、俯视图或观众遥测。特权 Critic 和
reward/evaluation ledger 仅用于训练。

在冻结门通过前，不声称取得了全风格 benchmark 成功或 official-test 结果。
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
