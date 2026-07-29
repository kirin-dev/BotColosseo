# Crystal Run: Extraction

[English](README.md)

先训练一个真正具备任务能力的第一视角游戏 Bot，再在不修改游戏规则、不给
Actor 隐藏状态的前提下，派生出三种玩家能够直观看懂的学习型风格。

![Crystal Run 四策略展示](docs/assets/extraction/showcase-board.png)

## 如何看视频

每段视频展示的都是**控制镜头的 Bot 的第一视角**。需要判断的是镜头 Bot 的
行为风格，而不是画面中出现的对手。右侧面板仅用于向观众解释行为，不会输入
策略。

| Bot | 重点观察 | 视频 | 证据层级 |
|---|---|---|---|
| **Strong** | 收集高价值物资，并将其成功带出 | [MP4](docs/assets/extraction/strong.mp4) | research selection |
| **Aggressive** | 连续造成 5 次 20 HP 伤害，击倒对手、拾取尸体缓存并撤离 | [MP4](docs/assets/extraction/aggressive.mp4) | directional Showcase |
| **Defensive** | 在风险下拉开安全距离，不依赖击杀并保护物资撤离 | [MP4](docs/assets/extraction/defensive.mp4) | validation demonstration |
| **Explorer** | 搜索多个物资区、升级已满背包、避免交战并撤离 | [MP4](docs/assets/extraction/explorer.mp4) | validation demonstration |

案例来自冻结的 240 局 validation 协议。渲染器只会对同一个预选案例尝试至多
5 次；实际轨迹没有完成对应因果链时，不会写出最终视频。

## 游戏任务

Crystal Run 是一个基于真实 ViZDoom 的紧凑型 1v1 搜打撤任务：

```text
搜索物资 -> 交战或脱离交战 -> 管理三格背包
         -> 到达任一撤离点 -> 将携带价值带出
```

- 单局 75 秒，两个中立撤离点。
- 30 秒后开放撤离，需要连续停留 3 秒。
- 双方均为 100 HP；每次有效命中固定造成 20 点伤害。
- 初始 30 发子弹，不换弹，不复活。
- 物资价值为 10、25、50；背包满三格后，高价值物资确定性替换最低价值物资。
- 死亡会丢失全部未带出物资，并生成可被对手拾取的尸体缓存。
- 击杀本身不计分，只有成功撤离带出的价值有效。

Actor 从 13 个宏动作中决策：静止、前后移动、左右平移、左右转向、前进转向
组合，以及原地/前进/转向攻击。满血对手需要 5 次有效命中才能被击倒。

## 一个 Base，三种学习风格

![策略架构](docs/assets/extraction/method.svg)

共享的 **Strong** 策略是视觉 CNN-GRU，通过脚本 Teacher 行为克隆、循环 PPO、
历史 checkpoint 与轻量 PFSP 对手采样训练。Aggressive、Defensive、Explorer
都是绑定同一个冻结 Strong Actor 哈希的小型有界残差 logit adapter：

```text
style_logits = strong_logits + max_delta * tanh(delta(features))
```

- **Aggressive：**有效发起交战，并形成
  命中 → 击杀 → 尸体缓存 → 撤离的完整转化。
- **Defensive：**低资源时脱离交战并保护有意义的物资，同时惩罚空背包蹲守和
  携带高价值时无谓交火。
- **Explorer：**覆盖有用物资区、升级背包并完成
  升级 → 撤离转化；不是只看移动距离，也不是只看撤离率。

三个风格绑定同一个冻结 Strong Actor 哈希。公开推理不包含规则式风格
governor。

## 公平观测

部署时 Actor 只能接收 84×84 第一视角灰度图，以及自身公开的血量、弹药、
背包、已带出价值、撤离状态、计时器和上一动作。Actor 看不到敌方血量/坐标、
自身世界坐标、深度图、目标标签、俯视图、特权协议状态或观众叠加层。

训练阶段可以使用特权 Critic 和 reward/evaluation ledger；公开推理只导出
Actor。

## 证据与边界

Strong checkpoint 通过冻结研究门：solo 撤离率 100%，脚本对手平均胜率
89.2%，validation 撤离率 94.6%，heldout-layout 撤离率 85.8%。

风格模型使用明确的证据层级：

- `research_selection`：完整通过冻结 validation 与 heldout 门；
- `directional_showcase`：产品方向和能力成立，同时公开研究门失败项；
- `validation_demonstration`：成对 validation 的方向、能力、反作弊和协议门
  通过，同时保留 heldout 失败披露。

Aggressive 的成对 validation 风格提升为 +0.101，任务保持率为 93.5%；其置信
区间下界和一个 heldout 对手保持门未通过。Defensive 的成对风格提升为 +0.006，
任务保持率为 96.3%，validation 撤离率变化为 +1.3 个百分点；其置信区间下界
未通过，heldout 撤离率变化为 −11.7 个百分点。Explorer 的成对风格提升为
+0.050，
任务保持率为 94.4%，validation 撤离率变化为 +0.8 个百分点；其 heldout
撤离率变化为 −17.5 个百分点。这些是产品 Showcase 证据，不是 official-test
结果。

在冻结门通过前，不声称取得了全风格 benchmark 成功或 official-test 结果。
候选选择阶段禁止访问 test。

机器可读的 [Showcase 审计](reports/extraction/showcase/audit.json) 绑定了每个
视频、案例、模型哈希、证据层级、公开失败项和渲染尝试记录。严格的全风格研究
发布与单次 official test 暂时保留为后续工作。冻结协议规定每个策略一次
400 局，official test 总计 1,600 局；当前尚未运行。

## 复现

使用 `botcolosseo` Conda 环境。完整训练、断点恢复、候选选择和视频生成命令见
[script.md](script.md)。

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

冻结任务规则与产品门见 [Plan.md](Plan.md)。

## 许可证

项目源码采用 MIT License。ViZDoom 与 Freedoom 保留各自许可证。本项目不
分发商业 Doom 素材，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
