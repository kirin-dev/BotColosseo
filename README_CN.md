# Crystal Run: Extraction

[English](README.md)

先训练一个真正有任务能力的视觉游戏 Bot，再在尽量保留能力的前提下，把它
塑造成玩家能够辨认的不同风格。

Crystal Run: Extraction 是一个基于真实 ViZDoom 的紧凑型 1v1“搜打撤”
研究产品：

```text
搜索物资 -> 交战或撤离交战 -> 管理三格背包
         -> 到达任一撤离点 -> 将携带价值带出
```

项目重点是产品可感知的智能体行为与严谨的工程闭环，而不是提出新的强化学习
算法。完整流程包含公平观测的循环策略、非对称训练、脚本对手、历史策略池、
轻量 PFSP，以及学习得到的残差风格适配器。

> 当前状态：v3 场景、训练栈、冻结评估协议、发布保护和真实端到端短程预检
> 已完成。完整 Strong 与风格实验是下一阶段；在冻结门通过前，不声称取得了
> benchmark 成功。

## 游戏规则

- 双方在同一张地图进行 75 秒对局，共有两个中立撤离点。
- 30 秒后开放撤离，需要连续停留 3 秒。
- 双方均为 100 HP，每次有效命中固定造成 20 点伤害。
- 初始 30 发子弹，不换弹，不复活。
- 背包有三个格子；物资价值为 10、25、50。
- 更高价值物资会确定性替换背包中最低价值物资。
- 死亡后生成可被拾取的尸体缓存。
- 击杀本身不计分，只有成功撤离带出的价值有效。

训练与常规 validation 使用 `base` 物资布局；heldout validation 和唯一一次
official test 使用已人工批准的 `heldout-a`：

![训练布局与 heldout 布局](docs/review/extraction-layout-review.svg)

## 四个 Bot

| 策略 | 期望可见行为 | 实现 |
|---|---|---|
| Strong | 稳定获胜、存活、收集价值并撤离 | CNN-GRU Actor、BC warm start、循环 PPO、脚本池、历史 checkpoint、PFSP |
| Aggressive | 主动制造有效交战，并把击杀后的缓存转化为撤离价值 | Strong 上的有界可学习 delta-logit adapter |
| Defensive | 在高风险下脱离战斗并保护物资，同时避免无物资蹲守 | Strong 上的有界可学习 delta-logit adapter |
| Explorer | 探索有用的新区域、升级背包，并最终完成撤离 | Strong 上的有界可学习 delta-logit adapter |

三个风格都绑定同一个冻结 Strong Actor 哈希。v3 推理过程不包含规则式风格
governor。

## 公平观测边界

部署时 Actor 只能看到：

- 84×84 第一视角灰度图像；
- 自身公开的血量、弹药、背包、已带出价值、撤离状态、剩余时间和上一动作。

Actor 看不到敌方血量、敌方坐标、自身世界坐标、俯视地图、深度图、目标标签、
特权协议状态或视频叠加层。只有训练阶段的 Critic、reward 和评估 ledger
可以使用特权状态；发布推理只导出 Actor。

## 训练与评估

```text
Strong Teacher 数据
       |
       v
行为克隆 -> 循环 PPO -> 历史 checkpoint + PFSP
       |
       v
唯一选定的 Strong Actor 哈希
       |
       +----------+-----------+
       v          v           v
   Aggressive  Defensive   Explorer
     adapter     adapter     adapter
```

候选选择阶段禁止访问 test：

- 每个策略 240 局成对 validation；
- 必要时每个策略增加 120 局 heldout-layout 评估；
- 冻结发布后，每个策略只运行一次 400 局 official test；
- official test 总计 1,600 局。

Strong 必须先通过全部能力门。每个风格必须至少保留 85% 的 Strong 成功案例，
撤离率下降不超过 10 个百分点，平均带出价值至少保留 85%，并且成对风格指标
及其 95% 置信下界都必须为正。

“阻止对手撤离率”仅作为 Aggressive 压迫性的辅助评估指标，不进入训练奖励，
也不要求击杀对手后才能撤离；搜打撤任务的胜负仍只由双方最终带出价值决定。

## 复现

项目 Conda 环境名为 `botcolosseo`。长实验、断点恢复、候选选择、official-test
锁和视频生成命令见 [script.md](script.md)。

短程验证：

```bash
conda activate botcolosseo
python -m pip check
python -m ruff check src tests scripts
python -m pytest tests/unit -q

python scripts/build_crystal_run_extraction.py \
  --check \
  --acc "$ACC_ROOT/build/acc" \
  --acc-include "$ACC_ROOT"
```

技术设计和发布门见 [Plan.md](Plan.md)。

## 许可证

项目源码采用 MIT License。ViZDoom 与 Freedoom 保留各自许可证。本项目不
分发商业 Doom 素材，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
