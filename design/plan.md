# LLM Post-Training 现象复现与分析项目计划
aa

> 目标：做一个 1k+ star 的开源项目，主题 = 后训练中的**遗忘 / 方差 / 涌现**，
> 用可复现的实验 + 量化分析讲清楚"为什么"，而不是又一个"怎么训"的框架。

## 1. 定位与差异化

- 定位一句话：**"Post-training, measured."**（后训练现象测量实验室）
- 不做的：新框架、新算法、多机分布式、>7B 模型、教程型内容（这些已被 TRL/OpenRLHF/verl/一堆教程占满）。
- 差异化：现有 repo 只展示"做出来了"；我们每个实验交付三件套：
  1. 一键复现命令（`uv run python -m experiments.xxx`）
  2. 一张主图（hero figure，直接放 README 顶部）
  3. 一句可引用的结论（带置信区间 / 多 seed 证据）
- 品牌原则：**预先注册指标，阴性结果也发布**。测量诚实本身就是卖点。
- 竞争格局（2026-07 核查）：
  - [SakanaAI/rl-razor-mnist](https://github.com/SakanaAI/rl-razor-mnist)（2026-03）已复现 razor 的 **MNIST 视觉**实验，"首个复现"卖点已不存在。
  - razor 结论已被二手媒体充分覆盖（MarkTechPost、Baseten continual-learning 综述等），正在变成常识。
  - 因此楔子不是"复现 razor"，而是：**LLM/token 尺度的测量 + 方差/seed 噪声（几乎没人系统做）+ 从业者可用的结论**（如 SFT vs RL 的 KL 预算参考）。发布定位 = "razor 在 LLM 尺度的对照实验 + 它没告诉你的方差问题"。

## 2. 目标与衡量

| 指标 | 目标 |
|---|---|
| GitHub star | 1000（里程碑：发布 1 → 100，发布 2 → 400，发布 3 → 1000） |
| 实验 | 3 个完整实验（遗忘 / 方差 / 涌现），各有主图 + 博客 |
| 复现成本 | toy 部分笔记本 CPU < 1 小时；模型部分单卡 4090 < 半天 |

期望值管理（诚实版）：star 数主要由发布曝光决定，无法排期保证。三次发布执行到位 → 三位数 star 是现实预期；1k 需要至少一次 HN 首页 / 大号转发级病毒曝光，作为拉伸目标而非验收标准。无论 star 如何，保底产出 = 3 篇可展示的分析作品 + 复现信誉。

## 3. 阶段总览

- Phase 0（~3 天）：仓库基础设施与打包
- Phase 1（~2 周）：CPU toy 实验 —— 遗忘（RL's Razor 复现）+ 方差
- 普通 8GB 显卡 1 天上手 LLM Post-Training！超简洁教程 + SFT/GRPO 对比 + RL 涌现现象
  - SFT和RL对模型改变的对比，看看用什么观察角度(具体的数据，指标等，可以通过sample)
    - 显式KL惩罚是防止reward hacking，on-policy采样主要是防止遗忘
  - 本身涌现出来的结构(在训练loss之外出现的结构都算，可以通过长短token，观察不同step，同一个问题的输出？；或者本身CoT?)
  - GRPO本身分母导致的bad example token序列变长的问题（这个估计训练量很大，做不起）

- Phase 2（~4-6 周）：Qwen2.5-0.5B 尺度复现 + 涌现探查（租单卡 GPU）
- Phase 3（贯穿 + 最后 1 周）：分发、运营、三次发布

---

## Phase 0：基础设施

- [ ] 修正 `REAMD.md` → `README.md`（当前文件名拼写错误）
- [ ] 确定公开 repo 名（见决策点 D1），写好 Apache-2.0/MIT License
- [ ] `uv` 管理依赖；`pyproject.toml` 最小依赖（torch / transformers / matplotlib / numpy，toy 阶段不引 trl/vllm）
- [ ] 目录约定：

  ```
  experiments/        # 每个实验一个子包，含 README + 预注册指标文件
    toy_forgetting/
    toy_variance/
    qwen_razor/
    qwen_emergence/
  src/                # 共享代码（模型、训练循环、指标、画图）
  assets/figures/     # 所有主图，README 引用
  ```

- [ ] 每个实验强制带 `metrics.md`：预注册假设、指标定义、seed 列表、验收标准
- [ ] GitHub Actions CI：toy 实验的冒烟测试（小配置跑 10 step 验证代码不挂）
- [ ] 出图规范：matplotlib 统一风格，脚本化生成到 `assets/figures/`，禁止手工截图

## Phase 1：CPU toy 实验

模型：2 层 tiny transformer（d_model=128，字符级，vocab≈20），CPU 单次训练 ≤ 几分钟。

### Exp 1A 遗忘：RL's Razor 定性复现

- 任务设计：预训练混合任务 A+B（如 A=加法，B=字符串逆序/模运算），模拟"base model 有先验能力"；微调阶段只在 B 上训练。
- 方法对比：
  - SFT：B 的参考答案上 teacher forcing
  - On-policy RL：GRPO 风格（group=8，组内归一化 advantage），reward=精确匹配，可加 KL 正则系数 β 扫描
- 核心协议（razor 的关键）：**在相同的 B 任务准确率处截停比较**（±1%），测 A 任务保持率与 KL(π_ft ‖ π_base)（在 B 输入上评估）。
- 扫描维度：学习率 × β × epoch 数，画出 (B 性能, KL) 与 (B 性能, A 保持率) 的 Pareto 前沿。
- 预注册假设：相同 B 性能下 RL 的 A 保持率显著高于 SFT；遗忘量与 KL 单调相关（Spearman ρ > 0.7）；SFT 大 LR 下出现"B 学会但 KL 爆炸、A 崩塌"的失效模式。
- 主图：`fig1_forgetting_vs_kl.png`（散点 + Pareto 前沿，RL vs SFT 两色）
- 与 SakanaAI/rl-razor-mnist 的关系：他们复现论文的 MNIST 视觉部分；我们覆盖 token 级语言任务并加多 seed 方差，后续上 0.5B——发布时明确引用并做定位区分，不抢"首个复现"叙事。

### Exp 1B 方差：seed 噪声有多大

- 同一配置跑 ≥16 seeds（SFT 与 RL 各一组）。
- 测：最终指标分布、训练曲线离散度、checkpoint 选择偏差（best-ckpt 与 mean-ckpt 的差距 = "checkpoint 彩票"）。
- 预注册假设：RL 组间方差显著大于 SFT；"常见幅度的 RL 提升"落在 seed 噪声 ±2σ 内。
- 主图：`fig2_seed_variance.png`（ridge/箱线 + 训练曲线束）

### 验收标准

- [ ] 两条一键命令各自 < 1 小时跑完并自动出图
- [ ] Razor 方向性结论复现（否则记录为阴性结果并分析原因，同样发布）
- [ ] 博客 #1（英文为主 + 知乎中文版）发布，README 更新主图 → **发布 v0.1**

## Phase 2：Qwen2.5-0.5B 尺度（租单卡 4090/24G，AutoDL 等）

预算粗估：0.5B 全量微调 bf16 + grad ckpt，单卡 24G 可行；单轮实验数小时，整阶段约 ¥300-800。

### Exp 2A 遗忘：razor 在真实模型上复现

- 持续学习设定：Stage 1 在任务 A（如 GSM8K 风格数学）训练；Stage 2 在任务 B（如 Countdown 或 IFEval 风格指令跟随）分别用 SFT / RL 训练。
- 测量：B 性能、A 保持率、通用能力探针（固定一小套 held-out eval）、B 分布上 KL。
- 协议同 toy：matched-B-performance 截停比较；LR / KL-β 扫描出 Pareto 前沿。
- 主图：`fig3_razor_qwen.png`

### Exp 2B 涌现：预注册候选指标（选 1-2 个，见决策点 D3）

- 候选 1：自我修正 token（"wait"/"不对"/重新检查类模式）出现频率随训练步的轨迹（Countdown/乘法任务，tiny-zero 范式）
- 候选 2：无长度惩罚下 CoT 长度的自发演化（压缩 or 膨胀？token 分布随训练的分位数曲线）
- 候选 3：pass@1 与 pass@k 的发散度（RL 是在"锐化"还是"扩展"推理边界）
- 阴性结果预案：若无涌现，发布带完整测量的阴性报告（标题如 "We tried to reproduce X at 0.5B — here's what actually happens"），仍符合品牌。

### Exp 2C 方差：主配置 ≥4-6 seeds，报告最终指标 spread 与长度曲线族

- 主图：`fig4_qwen_variance.png`

### 技术选型（决策点 D2）

- 倾向：自写最小 GRPO 训练循环（transformers + vLLM 采样），而非直接套 TRL——测量埋点（每步 KL、长度分布、token 频率）需要完全控制日志；代码本身也是"最小可读实现"卖点。
- [ ] 验收：2A 复现 razor 方向；2B 至少一个预注册指标有明确轨迹（或阴性报告）；2C 方差报告完成
- [ ] 博客 #2（razor 复现）、#3（涌现/方差）发布 → **发布 v0.2、v0.3**

## Phase 3：分发与运营（这是 1k star 的主战场，预留足量时间）

- [ ] README 英文优先（中文版链接）：顶部主图 + 一句话结论 + 一键 quickstart + 结果表 + citation 块
- [ ] toy 部分配 Colab/Kaggle notebook（10 分钟可跑），降低 star 门槛
- [ ] 三次发布节奏，每次 = Twitter/X 长帖（主图先行）+ HN (Show HN) + r/LocalLLaMA + 知乎/公众号
- [ ] 发布 2 时主动触达 RL's Razor 作者（X 上 @ + 邮件简报），争取转发/引用
- [ ] 投稿渠道：Hugging Face Daily Papers 相关讨论、机器之心、Papers with Code
- [ ] Issue 响应 < 24h；每个发布周每天看 GitHub Traffic 数据调整文案

## 时间线（约 10 周）

| 周 | 内容 |
|---|---|
| W1 | Phase 0 + toy 数据/模型打通 |
| W2-3 | Exp 1A/1B 跑完出图，博客 #1，**发布 v0.1** |
| W4-5 | 租卡，Exp 2A 跑通 + 扫描 |
| W6-7 | Exp 2B/2C |
| W8 | 博客 #2/#3，**发布 v0.2/v0.3** |
| W9-10 | README/文档打磨，社区运营，补实验 |

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 0.5B 无涌现 | 预注册指标 + 阴性报告照样发；必要时升到 3B（TinyZero 的证据在 3B，租金相应提高） |
| razor"首个复现"窗口已过（SakanaAI 已出 MNIST 版，结论已被媒体覆盖） | 叙事重心转向方差/seed 噪声与从业者可用结论；razor 部分定位"LLM 尺度对照实验"并引用 Sakana |
| toy 实验太"toy"没人看 | 主图必须一眼有冲击力；结论必须映射回真实痛点（razor/seed 噪声） |
| 分发执行不到位（恰是自身短板，全计划最高风险项） | 分发任务进 todo 与实验并列且优先排期；每次发布提前 1 周写帖子草稿；发布后 48h 专人盯评论区 |
| 范围蔓延（3 主题 × 2 尺度 = 6 块） | 每阶段只做列出的实验；新想法进 backlog 文件，发布后再议 |

## 决策点（待拍板）

- D1 公开 repo 名：沿用 `llm-posttraining`（描述性强）vs 更有记忆点的名字（如 `posttraining-lab`、`rl-razor`）？
- D2 Phase 2 训练栈：自写最小 GRPO 循环（推荐，测量可控）vs TRL 的 GRPOTrainer + 回调？
- D3 涌现候选选哪 1-2 个（候选 1/2/3）？
- D4 Phase 2 起步模型：0.5B-Instruct 直接上，还是直接 1.5B 提高涌现概率（租金更高）？

## 不做的事（防蔓延）

- 不做新训练框架 / 不卷 throughput
- 不做多机多卡 / 大于 7B 的模型
- 不写入门教程（市面上已经够多）
- 不在一个实验里塞第二个故事（一个实验一张图一句话）
