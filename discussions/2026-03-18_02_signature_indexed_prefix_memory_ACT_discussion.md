# Signature-Indexed Prefix Memory ACT

## 1. 方法定位

本文面向一类具有**隐式任务线索（task cue）**的长时序条件模仿学习任务。该类任务的关键信息往往仅在任务开始阶段短暂出现，而在后续执行过程中，当前观测逐渐失去对任务身份、阶段语义或目标分支的判别能力。此时，若策略仅依赖当前图像与当前状态进行动作预测，则容易在视觉局部相似但历史语义不同的状态下产生动作混淆。

为解决这一问题，我们提出 **Signature-Indexed Prefix Memory ACT**，简称 **SIPM-ACT**。该方法以 Action Chunking Transformer（ACT）为主干，在保持原始 chunk prediction 范式不变的前提下，同时引入两类**可流式更新、固定预算、长度无关**的历史摘要：

1. **状态前缀摘要**：通过 path signature 对状态历史进行前缀级压缩，编码系统从开始到当前的路径几何信息；
2. **视觉前缀记忆**：通过一个固定预算的视觉前缀记忆模块，对从开始到当前的视觉前缀进行在线压缩，持续保留起始阶段隐式 cue 对后续策略的影响。

方法核心不在于保存一段有限长度的历史窗口，也不在于从记忆库中检索若干局部事件，而在于构造一个对完整历史前缀进行**流式压缩与持续保持**的策略条件表示。

---

## 2. 核心思想

### 2.1 问题本质

我们关注的不是“最近几帧能否帮助当前决策”，而是：

> **任务开始阶段出现的隐式信息，能否被策略在较长时间后仍然保留下来，并持续影响动作生成？**

这意味着方法设计应满足以下三点：

- **全局前缀性**：历史表示描述的是“从开始到当前”的完整前缀，而不是若干离散事件；
- **流式一致性**：训练时和推理时都基于同一语义的在线历史摘要；
- **固定预算性**：历史表示的维度不随轨迹长度增长，保证部署效率与写作叙事统一。

### 2.2 为什么不用历史窗口

传统多帧方法通常截取最近 \(L\) 帧历史作为输入，但这种方式存在两个问题：

1. **窗口长度存在性能与实时性的折中**；
2. **超出窗口的早期关键信息会被彻底丢弃**。

对于“task cue 只出现在起始阶段”的任务，固定窗口并不是自然解法，因为它无法保证起始信息在后期仍然保留。

### 2.3 为什么不用事件记忆库

事件记忆库适合“中途关键事件需要被召回”的任务；而本文更关注“起始隐式线索决定整体执行分支”的场景。在这种设定下，使用可增长的 memory bank 和 top-\(r\) 检索并不自然，因为：

- 它强调的是**局部召回**，而非**全局保持**；
- 其容量会随时间增长，不符合本文强调的固定预算 streaming 叙事；
- 在纯专家示范的策略学习设定下，事件边界通常缺乏监督，容易引入额外建模复杂度。

因此，本文采用的是**前缀级压缩记忆**，而不是**事件级检索记忆**。

---

## 3. 方法总览

在每个决策时刻 \(t\)，策略接收以下输入：

- 当前视觉观测 \(I_t\)
- 当前低维状态 \(s_t\)
- 状态历史前缀 \(H_t = \{s_1, s_2, \dots, s_t\}\)
- 由状态前缀计算得到的路径签名摘要 \(g_t\)
- 由视觉前缀在线维护得到的视觉记忆 \(M_t^{vis}\)

策略输出长度为 \(K\) 的未来动作块：

\[
\hat{a}_{t:t+K-1} \sim \pi(\cdot \mid I_t, s_t, g_t, M_t^{vis})
\]

其中：

- \(g_t\) 表示“系统到当前为止走过了怎样的路径”
- \(M_t^{vis}\) 表示“系统到当前为止看到了什么，以及这些观察对当前任务身份意味着什么”

两者共同构成对完整历史前缀的双通道摘要。

---

## 4. 方法命名

**正式名称：** Signature-Indexed Prefix Memory ACT  
**简称：** SIPM-ACT

名称含义如下：

- **Signature-Indexed**：状态历史通过 path signature 编码，并参与视觉前缀记忆更新；
- **Prefix Memory**：视觉历史不是离散事件库，而是对完整前缀的固定预算压缩；
- **ACT**：保留原始 Action Chunking Transformer 的动作块预测框架。

---

## 5. 数学定义

### 5.1 状态前缀摘要

给定历史状态前缀：

\[
H_t = \{s_1, s_2, \dots, s_t\}
\]

定义其路径签名摘要为：

\[
g_t = \mathrm{Sig}(H_t)
\]

在实现上，\(g_t\) 可以有两种形式：

1. **真实 truncated path signature**
2. **结构相容的近似摘要**，例如基于状态增量的低阶矩、累积统计量或其他固定维度前缀嵌入

为了增强对局部新变化的敏感性，还可以引入增量摘要：

\[
\Delta g_t = g_t - g_{t-1}
\]

或采用更一般的增量签名贡献表示。

### 5.2 视觉前缀记忆

定义固定预算视觉前缀记忆为：

\[
M_t^{vis} \in \mathbb{R}^{B \times d}
\]

其中：

- \(B\) 为固定槽位数，通常为 1、2 或 4
- \(d\) 为 Transformer 隐空间维度

视觉前缀记忆通过流式更新函数维护：

\[
M_t^{vis} = \mathcal{U}_\theta\left(M_{t-1}^{vis}, \phi(I_t), s_t, g_t, \Delta g_t \right)
\]

其中：

- \(\phi(I_t)\) 为当前图像特征
- \(\mathcal{U}_\theta\) 为参数化更新器，可由 MLP、GRU、轻量 attention 或小型 SSM 实现

重要的是，\(M_t^{vis}\) 的维度与轨迹长度无关，因此它是严格意义上的 streaming prefix memory。

### 5.3 策略输出

在时刻 \(t\)，ACT 解码器基于当前 encoder memory 一次性预测未来动作块：

\[
\hat{a}_{t:t+K-1} = f_\theta(I_t, s_t, g_t, M_t^{vis})
\]

解码器仍采用固定数量的 query token 做 cross-attention，不改变原始 ACT 的 chunk prediction 机制。

---

## 6. 网络结构

## 6.1 总体结构

SIPM-ACT 保留 ACT 的主体设计，包括：

- 图像 backbone
- Transformer encoder
- Transformer decoder
- 可选的 VAE latent action prior

在此基础上新增两条历史分支：

1. **状态 signature 分支**
2. **视觉 prefix memory 分支**

### 6.2 编码器输入组成

编码器 memory 由以下 token 组成：

- 当前图像 token：由 backbone 提取
- 当前状态 token：由 \(s_t\) 线性投影得到
- signature token：由 \(g_t\) 经过 MLP 投影得到
- 可选的 delta-signature token：由 \(\Delta g_t\) 投影得到
- prefix memory token(s)：由 \(M_t^{vis}\) 直接作为固定数量的记忆 token 输入
- 可选的 VAE latent token

因此，编码器看到的不仅是当前观测，还包括两个固定预算的历史摘要通道。

### 6.3 视觉前缀记忆更新器

视觉前缀记忆更新器推荐采用以下三种之一：

#### 方案 A：GRU 风格更新器

对 \(B=1\) 的情况最直接：

\[
m_t^{vis} = \mathrm{GRU}\left(m_{t-1}^{vis}, [\phi(I_t), s_t, g_t, \Delta g_t]\right)
\]

优点是实现简单、训练稳定、易于部署。

#### 方案 B：多槽位交叉注意力更新器

对 \(B>1\) 的情况更合适：

- 以上一步 memory slots 作为 query
- 以当前图像特征、状态特征、signature 特征作为 key/value
- 通过一次轻量 cross-attention 完成写入与压缩

优点是表达能力更强，适合多对象、多阶段任务。

#### 方案 C：小型状态空间更新器

若强调 streaming 与长序列稳定性，也可用小型 SSM/LRU/Mamba 风格模块实现在线更新。

---

## 7. 训练方式

## 7.1 主损失

仍采用 ACT 的 chunk-level imitation learning 损失。给定示范动作块 \(a^*_{t:t+K-1}\) 与预测动作块 \(\hat{a}_{t:t+K-1}\)，基础重建损失写为：

\[
\mathcal{L}_{recon} = \sum_{k=0}^{K-1} \| \hat{a}_{t+k} - a^*_{t+k} \|_1
\]

并对 padding 位置进行掩码。

若启用 VAE，则总损失为：

\[
\mathcal{L} = \mathcal{L}_{recon} + \lambda_{KL}\mathcal{L}_{KL}
\]

## 7.2 推荐的辅助损失

为了更稳定地学习“前缀 cue 保持”，建议增加以下辅助项，但都不依赖人工事件标注。

### （1）Anchor Reconstruction Loss

让视觉前缀记忆去预测任务开始阶段的压缩表示，例如第一帧或前几帧的 anchor embedding：

\[
\mathcal{L}_{anchor} = \| \psi(M_t^{vis}) - z_{anchor} \|_2^2
\]

作用：鼓励 memory 持续保留起始 cue。

### （2）Past-Action / Past-Token Prediction

让策略头除了预测未来动作块，还预测部分过去动作 token：

\[
\mathcal{L}_{past}
\]

作用：增强历史保持能力，避免模型只依赖当前帧。

### （3）Signature Consistency Loss

让视觉前缀记忆与状态历史摘要在表示层面对齐：

\[
\mathcal{L}_{sig} = \| h(M_t^{vis}) - g_t \|_2^2
\]

或预测 \(\Delta g_t\)。

作用：强化视觉前缀历史与状态前缀历史的一致性。

### 7.3 最终损失

\[
\mathcal{L}_{total}
=
\mathcal{L}_{recon}
+
\lambda_{KL}\mathcal{L}_{KL}
+
\lambda_{anchor}\mathcal{L}_{anchor}
+
\lambda_{past}\mathcal{L}_{past}
+
\lambda_{sig}\mathcal{L}_{sig}
\]

实际实现中，可先从：

\[
\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{KL}\mathcal{L}_{KL}
\]

开始，再逐步加入一个最关键的辅助项。

---

## 8. 推理方式

SIPM-ACT 在推理阶段仍遵循 ACT 的 action chunking 机制。

每次前向时：

1. 获取当前图像 \(I_t\) 与状态 \(s_t\)
2. 更新状态前缀摘要 \(g_t\)
3. 更新视觉前缀记忆 \(M_t^{vis}\)
4. 预测未来长度为 \(K\) 的动作块
5. 执行前 \(M\) 步动作，其中 \(M \le K\)
6. 再次感知并滚动重规划

若启用 temporal ensembling，则对重叠动作进行加权融合；否则直接按队列消费动作。

这里的“streaming”含义非常明确：

- 不是内部有无限长隐状态展开
- 不是维护不断增长的记忆库
- 而是在固定预算下，对完整历史前缀持续进行流式压缩

---

## 9. 数据构造

SIPM-ACT 训练数据需要显式提供以下输入字段：

- `observation.image`
- `observation.state`
- `observation.path_signature`
- 可选：`observation.delta_signature`
- 可选：`observation.anchor_frame_embedding`

### 9.1 离线预计算部分

对于每条专家轨迹：

1. 读取状态序列 \(s_1, s_2, \dots, s_T\)
2. 按时间顺序计算每个时刻的 \(g_t\)
3. 若需要，再计算 \(\Delta g_t\)
4. 将其作为与图像、状态并列的输入写入数据集

### 9.2 在线推理部分

测试时：

1. 在线收集状态历史
2. 流式更新 \(g_t\)
3. 同步流式更新 \(M_t^{vis}\)

这样可保证训练与测试在输入语义上严格一致，不会出现“训练时看完整轨迹、测试时看不到”的分布不一致问题。

---

## 10. 方法优势

SIPM-ACT 相比标准 ACT 的优势可以概括为以下几点：

### 10.1 显式建模全局历史前缀

标准 ACT 主要基于当前观测做动作块预测；SIPM-ACT 显式加入对完整历史前缀的状态与视觉双通道压缩表示。

### 10.2 固定预算，不随时间增长

无论轨迹多长：

- path signature 维度固定
- visual prefix memory 槽位数固定

因此方法天然适合 streaming deployment。

### 10.3 与 ACT 高度兼容

方法不改变：

- chunk prediction 机制
- query-based decoder 结构
- VAE action prior 训练方式
- 推理时的滚动重规划方式

因此它是对 ACT 的低侵入式扩展，而非重构。

### 10.4 更契合“隐式起始 cue”任务

与事件记忆、有限窗口、多帧堆叠相比，SIPM-ACT 更自然地适配以下任务：

- 任务开始阶段决定执行分支
- 中后期当前观测局部相似
- 需要长期保留早期上下文
- 历史信息比局部变化更重要

---

## 11. 推荐基线

下面给出本文最应该包含的基线。它们的作用不是越多越好，而是要系统地回答“性能提升到底来自哪里”。

## 11.1 基础对照

### Baseline 1：Vanilla ACT
输入仅包含：

- 当前图像 \(I_t\)
- 当前状态 \(s_t\)

作用：最核心的基线，用于验证“仅看当前观测”的不足。

### Baseline 2：ACT + State Signature
输入包含：

- \(I_t\)
- \(s_t\)
- \(g_t\)

作用：验证仅引入状态前缀摘要是否足够。

这其实就是你当前 Streaming ACT 的核心版本。

---

## 11.2 视觉历史对照

### Baseline 3：ACT + Fixed History Window
输入包含：

- 当前图像
- 最近 \(L\) 帧历史图像
- 当前状态

作用：对比固定窗口式历史建模。

建议测试多个窗口长度，例如 \(L=4,8,16\)。

### Baseline 4：ACT + First-Frame Anchor
输入包含：

- 当前图像
- 第一帧 anchor 图像
- 当前状态

作用：验证“只保留任务开端静态视觉线索”是否已经足够。

这个基线非常关键，因为你的任务 cue 本来就出现在开始阶段。如果该基线已经很强，则说明你的方法必须证明自己超过“直接带着第一帧走”的简单方案。

### Baseline 5：ACT + Current Frame + Anchor + State Signature
输入包含：

- 当前图像
- 第一帧 anchor
- 当前状态
- \(g_t\)

作用：这是一个非常强的实用型基线，也最能逼近你的主方法。

如果主方法无法显著超越它，则“streaming prefix memory”的必要性会受到挑战。

---

## 11.3 前缀记忆对照

### Baseline 6：ACT + Visual Prefix Memory Only
输入包含：

- 当前图像
- 当前状态
- 视觉前缀记忆 \(M_t^{vis}\)

但不使用 \(g_t\)。

作用：验证视觉流式前缀摘要本身的贡献。

### Baseline 7：ACT + Signature-Indexed Prefix Memory（Full Model）
输入包含：

- 当前图像
- 当前状态
- \(g_t\)
- 可选 \(\Delta g_t\)
- \(M_t^{vis}\)

作用：完整方法。

### Baseline 8：ACT + Prefix Memory Without Signature Indexing
与完整方法结构相同，但 memory update 不使用 \(g_t\) 或 \(\Delta g_t\)。

作用：验证 path signature 在视觉前缀记忆更新中的作用，而不是仅仅多了一个 memory 模块。

---

## 11.4 辅助损失对照

### Baseline 9：Full Model w/o Anchor Loss
作用：验证起始 cue 保持正则项是否必要。

### Baseline 10：Full Model w/o Past-Token Loss
作用：验证过去 token 正则对长时依赖建模是否有帮助。

### Baseline 11：Full Model w/o Signature Consistency
作用：验证视觉与状态历史摘要的一致性约束是否带来提升。

---

## 12. 关键消融实验

建议至少做以下消融：

### 12.1 历史表示层面
- 无历史
- 仅状态 signature
- 仅 anchor frame
- anchor + signature
- 仅 visual prefix memory
- full model

### 12.2 前缀记忆容量
- \(B=1\)
- \(B=2\)
- \(B=4\)

作用：验证是否单 token 已足够，还是需要多槽位表达。

### 12.3 更新器类型
- MLP/GRU
- Attention update
- SSM-style update

### 12.4 Signature 形式
- 真实 truncated signature
- 近似 signature
- 仅 delta-signature
- global signature + delta-signature

### 12.5 任务长度泛化
训练较短轨迹，测试更长轨迹，验证 streaming prefix summary 的时长泛化能力。

### 12.6 起始 cue 干扰实验
故意扰动或遮挡起始 cue，观察各方法性能退化情况，验证模型是否真正依赖全局前缀历史。

---

## 13. 推荐实验故事线

本文的实验叙事建议按照下面顺序展开：

### 第一层：当前观测不够
先证明 Vanilla ACT 在视觉局部相似、但历史语义不同的状态下会失败。

### 第二层：状态历史有帮助但不够
再证明 State Signature 能解决一部分问题，但在纯视觉隐式 cue 或遮挡场景中仍然不足。

### 第三层：只带第一帧也不够
再证明 First-Frame Anchor 虽然有效，但无法处理执行过程中视觉变化、遮挡、视角漂移与状态演化。

### 第四层：真正需要的是前缀级视觉记忆
最后证明 SIPM-ACT 的 streaming prefix memory 能最稳定地保留起始 cue 并持续影响动作生成。

---

## 14. 适合的方法图描述

你后续画结构图时，建议采用如下布局：

### 左侧：输入流
- 当前图像 \(I_t\)
- 当前状态 \(s_t\)
- 历史状态前缀 \(H_t\)

### 中间上方：状态前缀摘要
- \(H_t \rightarrow g_t\)
- 可选 \(g_t \rightarrow \Delta g_t\)

### 中间下方：视觉前缀记忆
- 上一步 \(M_{t-1}^{vis}\)
- 当前视觉特征 \(\phi(I_t)\)
- 状态与 signature 条件
- 更新后得到 \(M_t^{vis}\)

### 右侧：ACT 主干
- 图像 token
- 状态 token
- signature token
- prefix memory token
- decoder query token
- 输出未来动作块

图中要突出两点：

1. **状态历史和视觉历史都采用前缀级流式压缩**
2. **视觉记忆不是增长式 bank，而是固定预算 memory slots**

---

## 15. 推荐论文表述模板

下面给出一段可直接写进论文方法部分的表述：

> We propose Signature-Indexed Prefix Memory ACT, a low-intrusion extension of Action Chunking Transformer for long-horizon imitation learning under implicit early-stage task cues. The key idea is to replace explicit history replay or growing event memories with two streaming prefix summaries of fixed budget: a state-prefix signature that encodes the geometry of the executed trajectory, and a visual prefix memory that incrementally compresses the entire visual prefix into a constant number of memory tokens. At each decision step, the policy conditions on the current observation together with these two prefix summaries to generate a future action chunk. In this way, the model preserves early task-relevant information throughout execution without increasing memory size with trajectory length.

---

## 16. 一句话总结

**SIPM-ACT 的本质不是“看更多历史帧”，而是“在固定预算下持续保留整个历史前缀中对当前决策仍然重要的信息”。**

---

## 17. 最终建议

如果后续要真正落地实现，我建议优先按以下顺序推进：

1. 先做 `Vanilla ACT`
2. 再做 `ACT + State Signature`
3. 再做 `ACT + First-Frame Anchor`
4. 再做 `ACT + Anchor + State Signature`
5. 最后做 `SIPM-ACT`

原因是这条路线最容易回答审稿人最关心的问题：

- 你到底是不是只需要起始帧？
- state history 和 visual history 哪个更重要？
- prefix memory 到底比简单 anchor 强在哪里？
- signature indexing 到底有没有必要？

如果这几组关系能讲清楚，这篇工作的主线就会非常稳。