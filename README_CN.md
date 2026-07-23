# Bot Colosseo

中文说明 · [English](README.md)

通过能力保持的策略塑形，构建目标导向、风格可控的视觉游戏 Bot。

Bot Colosseo 研究一个面向真实产品的问题：先训练具备任务能力的
fair-observation 视觉 Bot，再从同一个 Base checkpoint 派生玩家能够感知的
Aggressive、Defensive 与 Explorer 风格，同时把 Difficulty 作为独立控制维度。
完整技术方案见 [Plan.md](Plan.md)。

## Strong Base → Aggressive

![同一 validation case 上的 Strong Base 与 Aggressive](docs/assets/showcase/m4-base-vs-aggressive.gif)

| 冻结 validation 证据 | 结果 |
|---|---:|
| Strong Base 胜率 | 87.0% |
| Aggressive 胜率 | 89.0% |
| Aggressive engagement shift | +0.100 / 100 decisions |
| Skill Retention | 100.0% |
| 配对评测规模 | 200 局 |

Aggressive Bot 是从同一个 fair-observation Strong Base 派生的固定残差风格
checkpoint。它通过了预先定义的七项风格、安全和能力保持门禁：
engagement-shift 的 95% bootstrap 区间为 `[0.046, 0.171]`，有效攻击率为
26.7%，objective-chase rate 控制在 9.0%。

上方 GIF 是自动选择的定性 **validation** 案例，不是 official test 结果。
完整证据包括 [Strong Base 单局视频](docs/assets/showcase/m4-strong-base.mp4)、
[Aggressive 单局视频](docs/assets/showcase/m4-aggressive.mp4)、
[指标卡](docs/assets/showcase/m4-metrics.png)和
[hash-bound 发布清单](reports/showcase/m4/manifest.json)。

## 公平的 Easy / Normal / Hard 控制

![600 局配对 validation 上的难度表现](docs/assets/showcase/m5-difficulty.png)

同一组冻结 checkpoint 从 Easy 到 Hard 逐步减少推理限制。Strong Base performance
为 `0.820 → 0.878 → 0.955`，Aggressive 为
`0.830 → 0.890 → 0.955`。600 局正式 validation 的六项冻结门均通过，环境重试
和协议错误均为 0。Hard 是原生策略；Normal 增加一个 decision 的反应延迟；
Easy 增加两个 decision 的延迟，并且每两个 decision 才更新一次策略。

这是 Strong Base/Aggressive controller calibration 的通过结果，不是完整 M5
声明。Defensive 与 Explorer 仍需先通过风格门，再完成冻结的全风格 difficulty
扩展。详见[证据记录](docs/milestones/m5-difficulty.md)。

## 产品与技术主线

```text
Teacher demonstrations
        ↓
BC warm start → recurrent visual PPO
        ↓
historical opponents / PFSP capability anchor
        ↓
skill-preserving residual policy shaping
        ↓
Aggressive / Defensive / Explorer checkpoints
        ↓
independent Easy / Normal / Hard controller
        ↓
paired metrics + blind user evaluation
```

Actor 在推理时只接收 `84×84` 第一人称灰度画面、自身公开变量、比分、剩余
时间、是否持有核心和上一动作。敌方坐标、region ID、automap、depth、labels
等特权信息不得进入 Actor。Teacher、reward、Critic 和离线评测可以使用特权
状态，但其边界由类型、前向接口测试和 evidence audit 共同约束。

## 当前证据状态

| 里程碑 | 结果 | 证据边界 |
|---|---|---|
| M1 Crystal Run 与 Teacher | PASS | 5 项能力各 100/100，协议错误 0 |
| M2 真实同步 1v1 与初始 PPO | FAIL | 工程完整，但 PPO 未达到相对 BC +10pp |
| M3 historical/PFSP Strong Base | 未通过全部能力门 | 仅称 integrity-qualified capability anchor |
| M4 Aggressive | PASS | 200 局 validation，风格与 retention 七项门均通过 |
| M5 Defensive / Explorer / Difficulty | 进行中 | Difficulty 首个 600 局 block 通过；首版 200k PPO repair 已完成但未通过风格 smoke |
| M6 公开发布 | 待 M5 | 四策略展示、匿名用户评测与 checkpoint release |

M2 的 official 1,500 局配对 test 工程门完整且协议干净，但能力门没有通过：
PPO 胜率 77.0%，BC 为 75.2%，RandomLegal 为 34.4%；PPO objective rate
为 93.2%，BC 为 97.8%。仓库不会把工程闭环包装成 benchmark 成功。

M3 完成了 historical pool、PFSP、cross-play、候选选择与证据审计，但没有
通过全部冻结能力阈值，因此 200k checkpoint 仅作为后续风格塑形的 capability
anchor。

M4 已形成可信的垂直闭环。Defensive 与 Explorer 的 distillation 和首版 200k
closed-loop PPO 均完成，但仍分别未得到正向 protective-presence shift 和
flank route。完整负结果保留在
[Defensive 记录](docs/milestones/m5-defensive.md)和
[Explorer 记录](docs/milestones/m5-explorer.md)。已批准的 V2 使用 on-policy
Teacher 辅助和公平的 Explorer 内部路线模式，冻结评测门禁保持不变。

## 快速开始

```bash
conda env create -f env.yml
conda activate botcolosseo
python scripts/check_env.py
python -m pytest -v

ACC_PATH=/path/to/acc ACC_INCLUDE=/path/to/acc/source \
  python scripts/build_crystal_run.py --check

python scripts/smoke_crystal_run.py \
  --task moving_hit \
  --teacher aggressive_script \
  --record videos/m1-smoke.mp4 \
  --require-video
```

复现完整 M1 冻结门禁：

```bash
python scripts/evaluate_m1.py --split test --output reports/m1
```

它会运行 500 个真实 ViZDoom episode。M2 及后续训练、评测、审计、PID、日志和
长实验命令集中记录在 [script.md](script.md)；各里程碑的可读结论位于
[`docs/milestones/`](docs/milestones/)。

## 公平性与可复现性

- train / validation / test case 与随机种子隔离；
- paired evaluation 固定 opponent、seed、side 和 case；
- checkpoint、配置、scenario、ledger、summary 与媒体均记录 SHA-256；
- production publication 拒绝 dirty worktree、失败门禁和 test-derived 素材；
- 完整失败结果不删除，不在看到结果后修改冻结阈值；
- 演示素材明确标注 validation，与 official test 结果分开。

## 许可

Bot Colosseo 源代码采用 MIT License。ViZDoom 与 Freedoom 保留各自许可，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本仓库不分发商业 Doom
素材。
