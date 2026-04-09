# Signature-Indexed Slot Memory for Plug-and-Play Action Experts

## 1. 方法定位

本文提出一个新的可插拔记忆模块：**Signature-Indexed Slot Memory**，简称 **SISM**。其目标不是为某一个特定 policy backbone 设计专用历史分支，而是为任意 action expert 提供统一的、固定预算的、可流式更新的历史条件接口。

设任意动作专家记为

$$
\hat y_t = \mathcal E_\psi(x_t; c_t),
$$

其中 $x_t=(o_t,s_t)$ 为当前观测，$c_t$ 为额外条件。我们的核心贡献在于构造一个独立的前缀记忆模块

$$
c_t = \mathcal C_\theta(g_t,\Delta g_t,M_t),
$$

其中：

- $g_t$ 是状态前缀摘要；
- $\Delta g_t$ 是增量前缀摘要；
- $M_t \in \mathbb R^{K \times d}$ 是由多个 slot 组成的固定预算视觉前缀记忆。

与当前 GRU-style memory 不同，SISM 的关键新意不是“把所有历史压进一个 recurrent hidden state”，而是：

> **让状态前缀摘要显式充当 memory 写入路由器，对不同语义的历史信息进行 slot 级索引、分配与更新。**

---

## 2. 核心问题

我们关心的任务具有如下结构：

1. 早期阶段出现决定后续分支的隐式 cue；
2. 中后期局部观测高度相似，但历史语义不同；
3. 策略需要长期保持“我属于哪一条执行分支”的信息。

普通单 hidden state memory 的问题在于：

- 所有历史语义混写进同一个向量；
- 没有显式机制保证不同类型 cue 被分配到不同记忆子空间；
- 很难论证“state signature 真正起到了 indexed memory routing 的作用”。

因此我们希望把 memory 设计成：

$$
M_t = \{m_t^{(1)}, \dots, m_t^{(K)}\}, \qquad m_t^{(k)} \in \mathbb R^d,
$$

并让 $g_t,\Delta g_t$ 决定：

- 写哪个 slot；
- 写多少；
- 保留多少旧信息；
- 当前 expert 该读哪些 slot。

---

## 3. 状态前缀摘要

记状态前缀为

$$
H_t = \{s_1,\dots,s_t\}.
$$

定义状态前缀摘要

$$
g_t = \Phi_{\text{sig}}(H_t),
\qquad
\Delta g_t = g_t - g_{t-1}.
$$

在当前仓库中，$\Phi_{\text{sig}}$ 可直接实例化为 path signature。更一般地，它也可以替换为任意固定维前缀编码器。

然后将其投影为路由空间表示：

$$
r_t = \phi_r([g_t,\Delta g_t]) \in \mathbb R^{d_r}.
$$

这里 $r_t$ 的语义不是普通条件向量，而是 **memory addressing signal**。

---

## 4. Signature-Indexed Slot Memory

### 4.1 记忆状态

维护 $K$ 个 slots：

$$
M_t = [m_t^{(1)},\dots,m_t^{(K)}] \in \mathbb R^{K \times d}.
$$

每个 slot 被解释为：

- 一个可学习的历史语义缓存单元；
- 对某类前缀模式或任务子语义的固定预算压缩。

### 4.2 写入候选

先从当前观测构造候选写入向量：

$$
u_t = \phi_u([\,\phi_v(o_t),\phi_s(s_t),\phi_g(g_t),\phi_\Delta(\Delta g_t)\,]) \in \mathbb R^d.
$$

它表示“当前这一步值得写进 memory 的信息”。

### 4.3 路由打分

对每个 slot 计算 signature-indexed 路由分数：

$$
\alpha_t^{(k)}
=
\frac{
\exp\big(
q(r_t)^\top k_m(m_{t-1}^{(k)}) / \sqrt{d_r}
\big)
}{
\sum_{j=1}^{K}
\exp\big(
q(r_t)^\top k_m(m_{t-1}^{(j)}) / \sqrt{d_r}
\big)
},
$$

其中：

- $q(\cdot)$ 由 prefix signature 路由信号产生 query；
- $k_m(\cdot)$ 把已有 memory slot 映射到 routing key。

这一步使得：

> **slot 的选择不再仅由当前视觉内容决定，而是显式依赖“到目前为止走过了怎样的状态路径”。**

### 4.4 写门与保留门

进一步定义 slot 级写门：

$$
\gamma_t^{(k)} = \sigma\big(w_\gamma^\top [r_t, m_{t-1}^{(k)}, u_t]\big).
$$

然后进行更新：

$$
m_t^{(k)}
=
(1-\gamma_t^{(k)}\alpha_t^{(k)})\, m_{t-1}^{(k)}
+
\gamma_t^{(k)}\alpha_t^{(k)}\, \tilde u_t^{(k)},
$$

其中

$$
\tilde u_t^{(k)} = \phi_{\text{slot}}([u_t, m_{t-1}^{(k)}, r_t]).
$$

这个形式比 GRU-style hidden state 更具结构性，因为它显式分离了：

1. **路由**：写哪个 slot；
2. **门控**：写入强度多大；
3. **变换**：写入内容是什么。

---

## 5. 读出机制

memory module 需要向上游 action expert 提供统一条件。我们定义 read 权重：

$$
\beta_t^{(k)}
=
\frac{
\exp\big(
q_{\text{read}}(x_t,r_t)^\top k_{\text{read}}(m_t^{(k)})/\sqrt d
\big)
}{
\sum_{j=1}^{K}
\exp\big(
q_{\text{read}}(x_t,r_t)^\top k_{\text{read}}(m_t^{(j)})/\sqrt d
\big)
}.
$$

读出上下文为

$$
z_t^{\text{mem}} = \sum_{k=1}^{K} \beta_t^{(k)} v_{\text{read}}(m_t^{(k)}).
$$

最终统一条件接口写作

$$
c_t = \mathcal C_\theta(g_t,\Delta g_t,M_t,z_t^{\text{mem}}).
$$

根据 expert 类型，$c_t$ 可实例化为：

- token 序列；
- FiLM 参数；
- cross-attention KV；
- 全局拼接向量；
- hidden-state initializer。

---

## 6. 与现有 GRU-style memory 的本质差异

当前 GRU-style memory 的形式本质上是

$$
M_t = \mathrm{GRU}(M_{t-1}, z_t),
$$

其局限是：

- 历史被混合写入统一状态；
- slot 间没有显式角色分工；
- signature 只是 conditioning，而不是 memory addressing mechanism。

SISM 的真正创新点在于：

1. **signature-indexed routing**
   - $g_t,\Delta g_t$ 显式参与 slot 选择；
2. **slotized memory**
   - 不同语义历史可分配到不同 slot；
3. **expert-agnostic adapter**
   - 输出是统一条件接口，而不是 ACT 专用 token trick。

因此，论文中可以明确声称：

> 该方法不是“在 ACT 上附加一个 recurrent branch”，而是“提出一种由 prefix signature 显式索引的 slot-based streaming memory module”。 

---

## 7. 适配不同 policy backbone

### 7.1 ACT / Chunking Transformer

把 $c_t$ 转成：

- signature token；
- slot memory tokens；
- 可选 encoder FiLM。

### 7.2 Diffusion Policy

把 $c_t$ 转成：

- denoising network 的全局条件 embedding；
- 或作为 cross-attention context。

### 7.3 Autoregressive Policy

把 $c_t$ 转成：

- prefix tokens；
- 或增广 KV cache。

### 7.4 RNN / MLP Policy

把 $c_t$ 转成：

- hidden-state initializer；
- 或直接拼接到 policy input。

这使得方法天然符合 plug-and-play 叙事。

---

## 8. 训练目标

主任务损失保持与 expert 本身一致：

$$
\mathcal L = \mathcal L_{\text{expert}}.
$$

在此基础上，建议增加两类只服务于 memory 学习的辅助项。

### 8.1 Slot Diversity / Load Balancing

避免所有写入塌缩到同一个 slot：

$$
\mathcal L_{\text{balance}}
=
\sum_{k=1}^{K}
\left(
\frac{1}{T}\sum_t \alpha_t^{(k)} - \frac{1}{K}
\right)^2.
$$

### 8.2 Signature-Memory Consistency

鼓励读出的 memory 上下文与前缀摘要一致：

$$
\mathcal L_{\text{cons}}
=
\left\|
\phi_c(z_t^{\text{mem}}) - \phi_g(g_t)
\right\|_2^2.
$$

最终可用

$$
\mathcal L_{\text{total}}
=
\mathcal L_{\text{expert}}
+
\lambda_{\text{bal}}\mathcal L_{\text{balance}}
+
\lambda_{\text{cons}}\mathcal L_{\text{cons}}.
$$

---

## 9. 论文主贡献建议写法

如果采用这个方向，论文 contribution 可以写成：

1. 提出一个 **plug-and-play prefix memory module**，可作为统一插件接入不同 action experts；
2. 提出一个新的 **signature-indexed slot memory** 机制，使状态前缀摘要显式参与 memory routing，而非仅作为普通条件；
3. 在 `early-cue / long-horizon ambiguity` 任务上，系统证明该模块能跨 backbone 缓解现有 expert 的共同短板；
4. 在真实机器人任务上验证该模块的实际价值。

---

## 10. 最小可行实现路线

建议分三步做，而不是一次把 full paper 全压上去。

### Step 1: ACT 实例化

下面给出可直接用于实现的专业 AI coding prompt：

```text
你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作。当前仓库已经具备以下基础：

1. `streaming_act` 已支持：
   - `observation.path_signature`
   - `observation.delta_signature`
   - prefix-sequence training
   - visual prefix memory
   - signature-conditioned visual prefix memory
   - memory-conditioned encoder FiLM
2. 当前 visual prefix memory 的实现仍是 GRU-style recurrent hidden state，只是支持 `num_memory_slots` 个并行 GRUCell。
3. 当前目标不是做 full plug-and-play multi-backbone 版本，而是先完成 **ACT 实例化的 Signature-Indexed Slot Memory (SISM)**，验证该结构本身可训练、可评估、可消融。

开始前你必须先阅读并理解这些文件：

- `main/discussions/2026-03-18_02_signature_indexed_prefix_memory_ACT_discussion.md`
- `main/discussions/2026-04-08_03_signature_indexed_slot_memory_discussion.md`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/configuration_streaming_act.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/modeling_streaming_act.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/prefix_sequence.py`
- `main/scripts/train_policy.py`
- `main/scripts/eval_policy.py`
- `main/scripts/env/braidedhub_env.py`
- 如有必要，再看 `main/scripts/env/panda_route_env.py` 和 `main/scripts/env/metaworld_env.py`

任务目标：
在不破坏现有 `streaming_act` 训练/评估链路的前提下，把当前的 GRU-style visual prefix memory 升级为一个 **Signature-Indexed Slot Memory**，并保持它仍然作为 ACT expert 的历史条件插件使用。

这一步只做 ACT 实例化，不要提前抽象成跨 backbone 框架，不要顺手重构成通用 plugin API。

你要实现的核心结构：

1. 记忆状态不再只是“多个互相独立、结构相同的 GRU hidden state”，而是一个显式的 slot memory：
   - `memory_state`: `(B, K_slots, dim_model)`
   - 每个 slot 都表示一个可独立更新的前缀语义缓存
2. 当前前缀 signature 和 delta signature 必须显式参与 **slot routing**，而不仅仅只是作为普通输入拼接给 GRU：
   - 根据 `g_t` 和可选 `Δg_t` 产生 routing query
   - 对各个 slots 计算 routing logits / routing weights
   - routing 权重必须真正影响每个 slot 的写入强度
3. 更新机制必须是 **slot-wise gated update**，而不是直接调用现有 GRUCell 完事：
   - 至少显式区分：
     - 写入候选 `u_t`
     - 路由权重 `α_t^(k)`
     - 写门或更新门 `γ_t^(k)`
   - 最终更新形式应等价于：
     - 保留旧 slot
     - 按 `α_t^(k) * γ_t^(k)` 写入新信息
4. 读出时必须从多个 slots 聚合出用于 ACT 的 memory conditioning：
   - 可以是直接把所有 slots 作为 memory tokens 注入 decoder
   - 也可以再额外做 pooled readout 用于 encoder FiLM
   - 但要明确区分“slot state”与“pooled context”

实现要求：

1. 默认关闭，必须通过新增 config / CLI 开关控制；关闭时当前 `streaming_act` 行为必须完全不变。
2. 新结构必须在训练和评估阶段语义一致：
   - 训练时：从 prefix 序列扫描重建 slot memory
   - 评估时：在线递推更新 slot memory
3. 不允许未来泄漏；prefix 的最后一个 valid 元素必须仍然对应当前时刻。
4. 不允许破坏现有：
   - path signature 输入链路
   - delta signature 输入链路
   - prefix sequence training
   - visual prefix memory eval hooks
   - memory-conditioned encoder FiLM
5. 除非确有必要，不要修改 ACT 的 chunk prediction 机制，不要修改 decoder query 机制，不要重构整个 `streaming_act` 主干。
6. 如果你需要在当前 GRU-style memory 与新 SISM 之间并存，必须做到：
   - 两者互斥或清晰优先级
   - config 命名自解释
   - 老 checkpoint 在不开新特性时仍可正常加载

推荐新增配置项，命名可微调但语义必须完整：

- `use_signature_indexed_slot_memory: bool = False`
- `slot_memory_num_slots: int`
- `slot_memory_routing_hidden_dim: int`
- `slot_memory_use_delta_routing: bool`
- `slot_memory_use_softmax_routing: bool`
- `slot_memory_use_readout_pooling: bool`
- `slot_memory_balance_loss_coef: float = 0.0`
- `slot_memory_consistency_loss_coef: float = 0.0`

其中：

- 前两个辅助损失默认都应关闭；
- 这一步的最小目标是先把主结构跑通；
- 如果你实现了辅助损失，也必须确保默认关闭时当前 loss 行为与之前兼容。

你需要在代码里明确写清楚的细节：

1. slot memory 的 state shape
2. routing logits / routing weights 的 shape
3. candidate write vector 的 shape
4. slot-wise gate 的 shape
5. decoder 最终看到的是：
   - 全部 slots
   - 还是聚合后的 memory tokens
   - 还是两者都用
6. encoder FiLM 使用的是：
   - 所有 slots 的均值池化
   - attention readout
   - 还是单独的 pooled context

建议的最小数学落地形式：

- `r_t = route_proj([signature_t, delta_signature_t?])`
- `u_t = write_proj([visual_t, state_t, signature_t, delta_signature_t?])`
- `α_t = softmax(route_query(r_t) @ route_key(memory_prev)^T)`
- `γ_t = sigmoid(gate_proj([memory_prev, u_t, r_t]))`
- `memory_next = memory_prev + α_t * γ_t * (candidate - memory_prev)`

如果你采用其它等价形式，也可以，但必须满足“signature-indexed routing”这个方法学要求，而不是退化成“把 signature 再拼一遍给 GRU”。

验收标准：

1. 当 `use_signature_indexed_slot_memory=False` 时：
   - 训练 forward 行为与当前实现一致
   - 评估 rollout 行为与当前实现一致
2. 当 `use_signature_indexed_slot_memory=True` 时：
   - 模型前向确实使用 slot routing 与 slot-wise gated update
   - 训练路径能从 prefix sequence 重建 slot memory
   - 评估路径能在线维护 slot memory
3. 至少提供一个最小 smoke test：
   - 一次 batch forward
   - 一次 `braidedhub` 或 dataset eval rollout
4. 至少提供一个最小消融对照说明：
   - 当前 GRU memory
   - 新 SISM memory
   - 二者的 config 切换方式
5. 若某个环境暂时不支持新的 slot memory 路径，不允许静默 fallback，必须明确报错并解释原因。

请直接改代码。最终回复必须包含：

- 修改文件列表
- 新增配置项与 CLI
- SISM 的核心更新公式与对应代码实现位置
- slot state / routing / gate / readout 的 tensor shape
- 最小训练 / 评估 smoke test 命令
- 与当前 GRU-style memory 的差异
- 尚未覆盖的风险点
```

### Step 2: 通用 adapter

抽象出统一接口：

$$
c_t = \mathcal C_\theta(g_t,\Delta g_t,M_t).
$$

让 ACT 只成为一个 adapter 实例。

### Step 3: 第二、第三类 backbone

至少补：

- 1 个 diffusion-style backbone
- 1 个 autoregressive / recurrent backbone

这样 plug-and-play claim 才真正成立。

---

## 11. 最关键的实验清单

若这篇工作想冲 CoRL，最重要的实验不是“比 GRU 稍微强一点”，而是以下几组。

### 11.1 方法必要性

- Vanilla expert
- expert + signature only
- expert + GRU memory
- expert + slot memory without signature indexing
- expert + full SISM

这组实验回答：

> 新意到底来自 slot 化、来自 signature indexing，还是只是因为多了一个 memory？

### 11.2 通用性

至少三类 backbone：

- ACT
- diffusion
- autoregressive / recurrent

### 11.3 任务对准性

任务要系统控制：

- early cue 是否存在
- ambiguity 是否存在
- horizon 长度是否增加

### 11.4 可解释性

最好可视化：

- 不同步骤的 $\alpha_t^{(k)}$
- 不同任务分支的 slot 激活模式
- memory read/write 的时序变化

这会非常有助于证明“signature indexing”不是空话。

---

## 12. 一句话结论

如果你们要把当前工作从“合理工程增强”升级成“更像 CoRL 会买账的方法论文”，我认为最值得押注的方向就是：

> **把当前 GRU-style prefix memory 升级为一个由前缀 signature 显式索引的 slot-based streaming memory，并把它写成 expert-agnostic 的 plug-and-play module。**

这条线既保留了你们现有工作的基础，又能把“新意”从简单模块堆叠提升到更清晰的结构性方法贡献。
