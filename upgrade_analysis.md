# RegimeForge 升级方向分析

> 基于对完整代码库的深度审计，从**算法深度、工程质量、学术可发表性、实用价值**四个维度提出升级路线。

---

## 当前项目能力摘要

| 维度 | 现状 | 评级 |
|------|------|------|
| Agent 架构 | DQN / Oracle DQN / HMM+DQN / RCMoE-DQN | ★★★☆☆ |
| 环境 | 合成 Markov regime 市场，4 regime，3 action | ★★★☆☆ |
| 评估框架 | 多 seed + CI + t-test + LaTeX | ★★★★☆ |
| 可视化 | matplotlib 静态 + Rich TUI 实时 | ★★★★☆ |
| 实验工具 | smoke / full / ablation / OOD 套件 | ★★★★☆ |
| 代码质量 | 类型标注完善、模块化清晰 | ★★★★☆ |

项目在"研究工具台"层面已经比较完整。下面的升级方向按**投入产出比**从高到低排列。

---

## 一、算法层面升级（研究深度）

### 1.1 ⭐⭐⭐ Prioritized Experience Replay (PER)

**现状**：所有 agent 使用均匀随机采样的 `ReplayBuffer`。

**升级**：引入 TD-error 加权的优先级经验回放。

**理由**：
- 在 regime 切换的边界时刻，TD-error 往往最大——这些恰好是最需要学习的 transition
- PER 能加速 agent 对 regime 切换的适应，可直接作为一个消融实验点
- 实现成本低，只需修改 `dqn.py` 中的 `ReplayBuffer`

```diff
# 新增 SumTree + PrioritizedReplayBuffer
# 在 update() 返回 td_errors 用于更新优先级
# 加入 importance sampling weight 修正
```

**预估工作量**：~150 行代码

---

### 1.2 ⭐⭐⭐ Dueling DQN + Noisy Networks

**现状**：标准 MLP Q-Network。

**升级**：
- **Dueling**：将 Q(s,a) 分解为 V(s) + A(s,a)，regime 信息主要影响 V(s)，可以更好区分
- **NoisyNet**：用参数化噪声替代 ε-greedy，探索效率更高

**学术价值**：可以做一组 `DQN → Dueling → Noisy → Dueling+Noisy` 的消融实验，表明 RCMoE 的改进与 backbone 改进是正交的

---

### 1.3 ⭐⭐⭐⭐ 连续动作空间 + PPO/SAC 支持

**现状**：离散 3 动作（short / flat / long）。

**升级**：
- 支持连续动作（仓位比例 [-1, 1]），更贴近真实交易
- 引入 PPO 或 SAC 作为 policy gradient baseline
- RCMoE 架构可推广为 Mixture-of-Experts Actor-Critic

**学术价值**：极高。证明 MoE gating 机制在 policy gradient 方法中同样有效，大幅提升论文泛化性声明

---

### 1.4 ⭐⭐⭐ Attention-based Gate Network

**现状**：Gate 是简单的 2 层 MLP → softmax。

**升级**：
- 引入 temporal attention（过去 k 步的观测序列 → attention → gate weights）
- 让 gate 不仅能看当前状态，还能看短期历史来判断 regime

**理由**：真实 regime 识别需要时间上下文。当前 gate 只看单帧，信息有限。这也解释了为什么 HMM+DQN 能用滚动窗口捕获 regime——gate 应该也有类似能力。

```python
class TemporalGatingNetwork(nn.Module):
    """Gate with temporal attention over past k observations."""
    def __init__(self, obs_dim, n_experts, hidden_dim, context_len=8):
        ...
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
```

---

### 1.5 ⭐⭐ 层级式 MoE (Hierarchical MoE)

**现状**：flat MoE，所有 expert 平等竞争。

**升级**：
- 两层 gating：先分 macro-regime（risk-on / risk-off），再分 sub-regime
- 可以暴露更多可解释性

---

## 二、环境与数据层面升级

### 2.1 ⭐⭐⭐⭐ 真实金融数据接入

**现状**：纯合成市场。

**升级**：
- 用真实 ETF / 指数日线数据（SPY, QQQ, GLD 等）预处理后接入
- 使用 HMM / GARCH 拟合真实数据的 regime 参数，再作为 `regime_params` 注入
- 或直接 replay 真实价格序列，用 Hidden Markov Model 标注 regime label

**选项 A — 拟合后注入（推荐起步）**：
```python
# 从真实数据拟合 regime 参数
from hmmlearn import hmm
model = hmm.GaussianHMM(n_components=4)
model.fit(real_returns)
# 提取 drift/vol/transition 注入 TrainingConfig
```

**选项 B — 直接 replay**：
```python
class ReplayMarketEnv(SyntheticMarketEnv):
    """Replay real price series with regime labels from HMM."""
    def reset(self, seed=None):
        self.prices = self._load_real_prices(seed)
        self.regimes = self._infer_regimes(self.prices)
```

**学术价值**：从 synthetic 到 semi-realistic，是**发表所需的最关键一步**

---

### 2.2 ⭐⭐⭐ 多资产环境

**现状**：单资产交易。

**升级**：
- 引入 2-3 个相关资产，观测向量扩展为跨资产特征
- 动作空间变为资产配比向量
- Regime 对不同资产的影响不同（bull 时科技股涨，bear 时债券涨）

---

### 2.3 ⭐⭐ 非平稳环境 (Non-stationary)

**现状**：Regime 转移矩阵固定。

**升级**：
- 让转移矩阵随时间缓慢漂移（regime drift）
- 测试 agent 的在线适应能力
- 这是一个很好的 robustness 实验

---

## 三、工程与可用性升级

### 3.1 ⭐⭐⭐⭐ Web Dashboard（替换 TUI）

**现状**：Rich TUI 终端仪表盘（已经很棒）。

**升级**：构建一个 Web-based 可视化前端
- 实时训练曲线（WebSocket 推送）
- 交互式 regime embedding 散点图（可旋转、可 hover 查看 gate weight）
- 动态 policy surface heatmap
- 跨实验对比面板
- 实验管理界面（启动 / 停止 / 配置）

**技术栈建议**：FastAPI + WebSocket + Vite React / plain HTML + D3.js / Plotly

**理由**：Web UI 适合展示给审稿人、合作者，也适合录制 demo video。TUI 可以保留作为 headless 运行模式。

---

### 3.2 ⭐⭐⭐ 模型 Checkpoint 保存 / 加载

**现状**：checkpoint 只保存 metrics 和分析数据，**不保存模型权重**。

**升级**：
- 保存 `state_dict()` 到 checkpoint 目录
- 支持从 checkpoint 恢复训练
- 支持加载已训练模型进行评估

```python
# training.py checkpoint 时额外保存
torch.save(agent.online.state_dict(), checkpoint_dir / "model.pt")
torch.save(agent.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
```

**这是一个严重的功能缺失**。每次实验都要从零训练，无法复现或对比。

---

### 3.3 ⭐⭐⭐ WandB / TensorBoard 集成

**现状**：自定义 JSON metrics 写入。

**升级**：
- 支持 `wandb.log()` 或 `tensorboard SummaryWriter`
- 自动记录训练曲线、gate weight 分布、policy surface 等
- 利用 WandB Sweep 自动化超参搜索

```python
# 在 training.py 中可选集成
if self.config.use_wandb:
    wandb.log({"reward": episode_reward, "gate_accuracy": gate_acc}, step=episode)
```

---

### 3.4 ⭐⭐ 配置系统升级

**现状**：`TrainingConfig` dataclass + CLI args。

**升级**：
- 支持 YAML / TOML 配置文件
- 支持配置继承和覆盖
- 实验可完全由配置文件驱动

```yaml
# experiments/rcmoe_full.yaml
base: default
agent_type: rcmoe_dqn
n_experts: 4
episodes: 2000
seeds: [17, 42, 123, 456, 789]
```

---

### 3.5 ⭐⭐ 并行实验执行

**现状**：实验串行执行。

**升级**：
- 使用 `multiprocessing` 或 `Ray` 并行执行不同 seed 的训练
- 将 full suite（7 方法 × 5 seeds = 35 runs）的执行时间缩短 5-10 倍

---

## 四、学术可发表性升级

### 4.1 ⭐⭐⭐⭐ 统计分析增强

**现状**：有 Welch t-test + Cohen's d + 95% CI。

**升级**：
- 加入 **Bootstrap Confidence Interval**（不依赖正态假设）
- 加入 **Wilcoxon signed-rank test**（非参数检验，5 seeds 时更可靠）
- 加入 **Friedman test + Nemenyi post-hoc**（多方法同时比较的标准做法）
- 加入 **Effect Size 的 Bayesian 分析**

```python
from scipy.stats import wilcoxon, friedmanchisquare
# Bayesian A/B testing via PyMC
```

**理由**：5 个 seed 的样本量很小，t-test 的 power 极低。顶会审稿人会质疑统计显著性。

---

### 4.2 ⭐⭐⭐ 可解释性分析套件

**现状**：Gate weight 分析 + activation heatmap + linear probing + t-SNE。

**升级**：
- **SHAP / Integrated Gradients for Gate**：哪些特征最影响 gate 的 routing 决策？
- **Expert Counterfactual**：如果强制使用 expert k，reward 会如何变化？
- **Gate Decision Boundary Visualization**：在特征空间中画出 gate 的决策边界
- **Regime Transition Analysis**：agent 在 regime 切换点的行为滞后有多大？

---

### 4.3 ⭐⭐⭐ 自动论文图表生成

**现状**：matplotlib 静态图 + LaTeX 表格。

**升级**：
- **Figure auto-layout**：一键生成论文标准的 multi-panel figure
- **Camera-ready 格式**：自动设置 font size, figure size 为 NeurIPS / ICML 标准
- **附录自动生成**：完整实验配置 + 所有 seed 的详细结果

```python
def generate_paper_figures(results, output_dir, style="neurips"):
    """Generate all figures needed for a paper submission."""
    ...
```

---

### 4.4 ⭐⭐ Reproducibility 工具包

**现状**：有 seed 控制。

**升级**：
- 环境快照：`pip freeze > requirements.txt`、Python 版本、CUDA 版本
- 配置 hash 绑定：每个实验结果绑定到完整配置的 hash
- Docker image for exact reproduction

---

## 五、推荐的升级优先级

基于投入产出比，建议按以下 Phase 推进：

### Phase 1：补关键短板（1-2 天）
1. **模型权重保存 / 加载**（3.2）—— 阻塞了所有后续实验
2. **PER 经验回放**（1.1）—— 直接提升训练质量

### Phase 2：提升学术强度（3-5 天）
3. **真实数据接入**（2.1 选项 A）—— 论文可发表性的分水岭
4. **统计分析增强**（4.1）—— Bootstrap CI + Wilcoxon
5. **Temporal Attention Gate**（1.4）—— 核心算法改进

### Phase 3：扩展研究范围（5-7 天）
6. **连续动作 + PPO/SAC**（1.3）—— 大幅扩展泛化声明
7. **可解释性分析套件**（4.2）—— SHAP + counterfactual
8. **Web Dashboard**（3.1）—— 展示 + 交互

### Phase 4：完善工程（按需）
9. WandB 集成（3.3）
10. YAML 配置（3.4）
11. 并行执行（3.5）
12. 多资产环境（2.2）

---

## 总结

> [!IMPORTANT]
> **最高优先级的两件事：** 模型权重保存（已是功能缺陷）和 Prioritized Replay（性价比最高的算法改进）。
>
> **如果目标是发表论文：** 真实数据接入 + 统计增强 + Temporal Attention Gate 是必做项。
>
> **如果目标是做成工具产品：** Web Dashboard + WandB 集成 + 并行执行是重点。

选择你最感兴趣的方向，我可以展开为详细的实现计划。
