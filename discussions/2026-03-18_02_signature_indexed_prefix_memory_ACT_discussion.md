# Coding Prompts

## 0. 共享实现约定

以下 prompts 是修订版。它们保留原有实验路线，但补齐了工程落地时最容易出问题的实现契约。若各 prompt 内未特别说明，统一遵守以下约定：

1. 所有新特性默认关闭；关闭时，现有 `act`、现有 `streaming_act`、现有 checkpoint 加载行为必须保持不变。
2. 训练和评估必须语义一致，不允许训练时看完整未来、评估时只看在线历史。
3. 所有新增输入都必须走完整链路：dataset export -> `meta/info.json` -> `meta/stats.json` -> preprocessing -> model forward -> online eval。缺任一环都不算完成。
4. 所有新增特性都必须通过明确的 config 和 CLI 控制，命名要自解释，不要用模糊短名。
5. 所有 shape 约定都必须在代码里通过断言或显式报错检查，而不是依赖隐式广播。
6. 新增 prefix 序列时，统一按时间从旧到新排列，最后一个 valid 元素就是“当前时刻”。禁止未来泄漏。
7. prefix 序列统一采用右侧 padding，并显式提供 `prefix_mask` 或语义等价的 valid-length 信息；`True` 表示有效位置。
8. 新增非图像 token 统一先投影到 `dim_model`，再以 `(batch, 1, dim_model)` 或 `(batch, num_slots, dim_model)` 的形式注入 encoder memory。不要把历史特征硬拼到图像通道维。
9. 除非 prompt 明确要求，否则不要重构 ACT 主干，不要修改 chunk prediction 机制，不要顺手引入额外方法学扩展。
10. 每一步都要给出至少一个最小 smoke test，优先使用真实 train/eval 命令而不是伪代码。

推荐统一的数据接口如下；如果代码库已有更自然的命名，也可以采用等价命名，但必须在最终回复中明确写出映射关系：

- 单时刻输入：
  - `observation.state`: `(B, state_dim)`
  - `observation.path_signature`: `(B, signature_dim)`
  - `observation.delta_signature`: `(B, signature_dim)`，仅在启用时存在
  - `observation.anchor_image` 或等价 anchor 字段
- prefix 序列输入：
  - `observation.prefix_state`: `(B, T_prefix, state_dim)`
  - `observation.prefix_path_signature`: `(B, T_prefix, signature_dim)`
  - `observation.prefix_delta_signature`: `(B, T_prefix, signature_dim)`，仅在启用时存在
  - `observation.prefix_mask`: `(B, T_prefix)`，`True` 表示有效
  - `observation.prefix_images.<camera>` 或等价 prefix 图像字段：`(B, T_prefix, C, H, W)`

若某个环境暂时不支持某一步，请显式报错并解释原因，不允许静默 fallback。

- [ ] Prompt 1：实现 First-Frame Anchor 基线

```
你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作。当前仓库已经实现了 `Vanilla ACT` 和 `ACT + State Signature`，其中 `ACT + State Signature` 对应现有 `streaming_act`。你的任务是在不重构主干的前提下，以最小侵入方式新增 first-frame anchor 输入链路，得到两个可运行基线：

1. `ACT + First-Frame Anchor`
2. `ACT + First-Frame Anchor + State Signature`

你需要先详细阅读并理解`main/discussions/2026-03-18_02_signature_indexed_prefix_memory_ACT_discussion.md`的`共享实现约定`部分，这是基础。

开始前先阅读这些文件：
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/configuration_act.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/modeling_act.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/processor_act.py`
- `main/scripts/train_policy.py`
- `main/scripts/eval_policy.py`
- `main/scripts/env/braidedhub_env.py`
- 如有必要，再看 `main/scripts/env/panda_route_env.py` 和 `main/scripts/env/h_shape_env.py`

任务定义：
anchor 的语义被严格定义为“episode reset 之后、执行第一步动作之前采集到的第一帧视觉观测”。它不是任意历史帧，也不是滑动窗口中的旧帧。

实现要求：
1. 新增显式开关，例如 `use_first_frame_anchor`，默认关闭。
2. anchor 训练语义和评估语义必须完全一致：
   - 训练时：dataset export 阶段把 first-frame anchor 写入数据集；
   - 评估时：rollout 开始后缓存第一帧，并在整个 episode 中重复复用。
3. anchor 最终必须作为额外的 encoder memory token 注入，而不是与当前图像做 channel concat。
4. 优先使用“原始 anchor 图像 -> 共享 backbone 编码 -> 压缩成单个 anchor token”的方案。
   - 若代码库限制导致必须存储离线 anchor embedding，也可以接受；
   - 但必须在训练和评估都采用同一表示，并在最终回复中说明原因。
5. 若使用原始 anchor 图像，必须明确 anchor token 的构造方式，例如：
   - 共享当前图像 backbone；
   - 再通过 pooling / projection 得到一个 `(B, dim_model)` anchor 表示；
   - 最终注入为 `(B, 1, dim_model)` token。
6. `act` 和 `streaming_act` 都要支持该特性；关闭时两条路径的行为必须和当前完全一致。
7. 数据集 metadata、stats、feature schema、preprocessor 都要同步更新，确保该特征不会绕过归一化和类型检查。
8. 不能删除或破坏现有 `observation.path_signature` 流程。
9. 优先先让 `braidedhub` 跑通；其它环境如果短时间内不支持，要显式报错并说明原因。

需要你在代码里明确写清的细节：
- anchor 数据字段的名字
- anchor 在 dataset 中存的是原图还是 embedding
- anchor token 的 shape
- anchor token 注入 encoder 的位置
- rollout 中 anchor 的缓存生命周期

边界条件：
- 保持 checkpoint 向后兼容，老模型在不开 anchor 时仍可加载。
- 不要引入不断增长的历史缓存。
- 不要修改 ACT 的 chunk prediction 机制。
- 不要把这一步扩展成 visual prefix memory。

验收标准：
1. `--policy act` 且不启用 anchor 时，训练与评估路径和当前行为一致。
2. `--policy streaming_act` 且不启用 anchor 时，训练与评估路径和当前行为一致。
3. 启用 anchor 时，模型前向图中确实多出一个 anchor token，并能完成一次 batch forward 和一次 rollout。
4. `braidedhub` 至少提供一个最小 smoke test，证明 train/eval 参数链路打通。
5. 若某环境尚未支持 anchor，不允许静默忽略，必须给出明确错误信息。

请直接改代码。最终回复必须包含：
- 修改文件列表
- 新增配置项和 CLI
- anchor 数据字段与 tensor shape
- 最小训练 / 评估 smoke test 命令
- 尚未覆盖的风险点
```

- [ ] Prompt 2：补齐 prefix-sequence 训练基础设施

```
你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作。当前 `streaming_act` 只实现了“当前时刻输入 + path signature token”，但讨论中的 SIPM-ACT 需要训练可在线更新的 prefix memory。现有单时刻训练样本不足以学习 `M_t = U(M_{t-1}, ...)`。你的任务是先补齐 sequence-aware 的 prefix 训练基础设施，但这一步先不要真正实现 visual prefix memory。

你需要先详细阅读并理解`main/discussions/2026-03-18_02_signature_indexed_prefix_memory_ACT_discussion.md`的`共享实现约定`部分，这是基础。

开始前先阅读：
- `main/scripts/train_policy.py`
- `main/scripts/eval_policy.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/configuration_act.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/modeling_act.py`
- `main/scripts/env/panda_route_env.py`
- 若训练数据读取逻辑不清楚，再继续追踪 dataset 构造、导出与 collate 代码

任务目标：
为后续 prefix memory 提供训练所需的数据和接口，使一个训练样本不再只包含当前帧，而是可以包含“从 episode 开始到当前时刻”的 prefix 子序列。

必须明确的数据契约：
1. 保留现有单时刻训练路径作为默认行为；新的 prefix 路径必须由显式开关启用，例如 `use_prefix_sequence_training`。
2. prefix 样本必须按时间从旧到新排列，并满足：
   - 最后一个 valid prefix 元素对应当前时刻；
   - 不包含任何未来帧；
   - 支持右侧 padding 和显式 mask。
3. 新路径下，每个训练样本至少返回：
   - prefix 图像序列
   - prefix 状态序列
   - prefix path_signature 序列
   - prefix 有效位 mask 或 valid-length
   - 当前时刻对应的监督动作 chunk
4. 推荐新增如下字段或语义等价字段：
   - `observation.prefix_state`
   - `observation.prefix_path_signature`
   - `observation.prefix_mask`
   - `observation.prefix_images.<camera>`
5. 新增长度预算控制参数，例如：
   - `prefix_train_max_steps`
   - `prefix_frame_stride`
   - 如有必要，再加 `prefix_pad_value` 或等价参数
6. 模型侧要为后续 prefix memory 预留 forward 接口，例如允许 forward 接收 prefix sequence tensors 和 mask；
   - 但这一步先不要引入任何可学习 memory 模块；
   - 也不要把完整 prefix 直接展平成最终的 encoder token 方案。
7. 代码里必须包含 shape 断言和错误信息，至少覆盖：
   - mask 与 prefix 长度不一致
   - prefix 最后一个 valid 位置缺失
   - 字段缺失
   - `state_dim` / `signature_dim` / 图像 shape 不匹配

环境范围：
- 优先先在 `panda_route` 打通。
- 若其它环境短期内不补齐，至少不要被破坏，并给出清晰报错。

边界条件：
- 这一步的本质是“数据和接口建设”，不是“主方法结果实现”。
- 不要加入 anchor loss、memory loss 或其它辅助任务。
- 不要破坏现有 `Vanilla ACT` 和 `ACT + State Signature` 的训练可用性。

验收标准：
1. 默认配置下，当前训练行为不变。
2. 开启 prefix-sequence 模式后，dataset -> preprocessor -> collate -> model forward 的 tensor shape 全链路打通。
3. 至少有一个最小 smoke test，能在 `panda_route` 上跑通一个 batch 的前向或极短训练。
4. 对 prefix 长度超限、缺失字段、shape 不匹配等情况有明确报错。

请直接改代码。最终回复必须包含：
- 旧的单时刻模式与新的 prefix 模式如何区分
- prefix 数据字段、shape、padding/mask 语义
- 修改文件列表
- 新增配置项和 CLI
- 一个最小 smoke test 命令
- 后续实现 visual prefix memory 时将复用哪些接口
```

- [ ] Prompt 3：实现 Visual Prefix Memory Only 基线

```
你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作。现在仓库应已具备 prefix-sequence 训练接口。你的任务是在此基础上实现一个最小可用的 `ACT + Visual Prefix Memory Only` 基线，用来回答“仅视觉前缀记忆本身是否有效”。这一步不要做 signature indexing，也不要加复杂辅助损失。

你需要先详细阅读并理解`main/discussions/2026-03-18_02_signature_indexed_prefix_memory_ACT_discussion.md`的`共享实现约定`部分，这是基础。

开始前先阅读：
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/configuration_act.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/modeling_act.py`
- `main/scripts/train_policy.py`
- `main/scripts/eval_policy.py`
- `main/scripts/env/panda_route_env.py`
- 以及你上一阶段加入的 prefix-sequence 相关代码

任务定义：
实现一个 fixed-budget 的视觉前缀记忆模块，只依赖 prefix 视觉信息和必要的状态信息，对完整 prefix 做在线压缩，并把最终 memory 作为额外 token 注入 ACT encoder。

实现要求：
1. 新增显式开关，例如 `use_visual_prefix_memory`，默认关闭。
2. 先只实现最小版本：`num_memory_slots=1`。
   - 不要再用 `B=1` 这个写法，以免和 batch size 混淆。
3. memory 状态的标准表示定义为：
   - 内部更新态：`(batch, dim_model)` 或 `(batch, 1, dim_model)`；
   - encoder 注入态：`(batch, 1, dim_model)`。
4. memory 更新器先采用 GRU 风格更新，不要一开始做多槽位 attention。
5. 更新器输入只允许依赖：
   - prefix 视觉特征
   - 必要的当前/对应时刻状态特征
   - prefix mask
   这一步不要使用 `g_t`，不要使用 `delta_signature`。
6. 训练时，必须通过扫描 prefix 序列来重建 `m_t`：
   - 从零初始化或明确声明的可学习初始态开始；
   - 按时间顺序仅使用 valid prefix 步更新；
   - 最终得到当前时刻的 memory token。
7. 评估时，memory 必须随 episode 在线更新，并在 `policy.reset()` 时清空。
8. 更新后的 memory 必须作为额外 encoder memory token 注入主干，而不是作为 loss-side feature。
9. 默认关闭时，当前 `streaming_act` 行为必须完全不变。
10. 优先先支持 `panda_route`，其它环境至少不要被破坏。
11. 这一步不要顺手加 anchor、不要顺手加辅助损失、不要顺手做 signature indexing。

需要你在代码里写清楚的细节：
- memory 初始态是什么
- 训练时 memory 如何从 prefix 序列扫描得到
- 评估时 memory 在哪一步更新
- `policy.reset()` 清空哪些缓存
- memory token 注入 encoder 的位置

边界条件：
- 不要让模型依赖未来 prefix 帧。
- 不要在 config 里加入一堆暂时不用的复杂参数。
- 不要把 memory 存成整个 prefix 序列；memory 必须是固定维度状态。

验收标准：
1. `use_visual_prefix_memory=false` 时行为与当前完全一致。
2. `use_visual_prefix_memory=true` 时，模型前向能接收 prefix 输入并输出动作 chunk。
3. 评估 rollout 中 memory 会逐步更新，而不是固定不变。
4. `policy.reset()` 后 memory 被清空，不会跨 episode 泄漏。
5. 至少提供一个最小训练 smoke test 和一个最小评估 smoke test。

请直接改代码。最终回复必须包含：
- visual prefix memory 的状态表示
- 更新发生在训练哪里、评估哪里
- 修改文件列表
- 新配置项
- 最小训练 / 评估 smoke test 命令
- 仍然简化了哪些地方
```

- [ ] Prompt 4：实现完整 Signature-Indexed Prefix Memory ACT

```
你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作。当前目标是在已有 `ACT + State Signature` 和 `Visual Prefix Memory Only` 基础上，实现讨论稿中的完整 `Signature-Indexed Prefix Memory ACT`，即让 visual prefix memory 的更新显式依赖状态前缀摘要 `g_t`，以及可选的 `delta_signature`。这是主方法实现，不再是基线。

你需要先详细阅读并理解`main/discussions/2026-03-18_02_signature_indexed_prefix_memory_ACT_discussion.md`的`共享实现约定`部分，这是基础。

开始前先阅读：
- `main/discussions/2026-03-18_02_signature_indexed_prefix_memory_ACT_discussion.md`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/configuration_act.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/modeling_act.py`
- `main/scripts/train_policy.py`
- `main/scripts/eval_policy.py`
- `main/scripts/env/panda_route_env.py`

任务目标：
实现完整的 SIPM-ACT 主干，包含：
1. state signature token
2. optional delta-signature token
3. signature-indexed visual prefix memory update
4. prefix memory token 注入 encoder memory

实现要求：
1. 新增 `observation.delta_signature` 的全链路支持：
   - dataset 导出
   - metadata / info / stats
   - preprocessing
   - training input
   - online evaluation
2. `delta_signature` 必须被严格定义为 `g_t - g_{t-1}`。
   - 第一个 valid prefix 步的 `delta_signature` 必须明确约定，推荐使用全零向量；
   - 训练和评估必须采用同一规则。
3. visual prefix memory 更新器必须显式接收 `g_t`，并可选接收 `delta_signature`。
4. 仍先支持 `num_memory_slots=1` 的单槽位 memory，但代码结构必须为未来 `num_memory_slots>1` 留清晰接口。
5. 训练时必须通过扫描 prefix 序列显式构建当前 memory；评估时必须在线逐步更新，不能偷看未来。
6. 模型 encoder memory 中要清晰区分并文档化以下 token：
   - 当前图像 token
   - 当前状态 token
   - signature token
   - optional delta-signature token
   - prefix memory token
   - optional latent token
7. 对新增 1D token 的位置编码或类型编码策略要明确：
   - 可以用单独 learnable slot，也可以用零位置编码；
   - 但必须在实现和最终回复中写清楚，不要含糊。
8. 所有新增特性都必须由 config 控制，默认关闭，保证老 checkpoint 和老训练脚本不坏。
9. 优先先让 `panda_route` 跑通；如果 `h_shape` / `braidedhub` 也能兼容更好。
10. 不能把这一步退化成“只是多加一个 delta-signature token”，核心是 visual memory update 被 signature conditioning。
11. 不要在这一步加入辅助损失，先把主结构跑通。

建议你在代码中提供一个清晰的 memory updater 接口，语义类似：
- 输入：`memory_prev, visual_t, state_t, signature_t, delta_signature_t, valid_t`
- 输出：`memory_t`
接口名不强制一致，但语义必须清楚。

设计约束：
- 这是 fixed-budget streaming prefix memory，不是增长式 bank。
- 这是 prefix compression，不是 sliding event retrieval。
- 不能破坏 ACT 的 chunk prediction 机制。
- 训练和推理必须共享一致的更新语义。

验收标准：
1. full model 的前向图中，确实存在 signature-conditioned memory update，而不是空挂 config。
2. `delta_signature` 在离线训练和在线评估中定义一致。
3. 默认配置下，老行为保持不变。
4. full model 至少有一个最小 batch forward smoke test 和一个最小 rollout smoke test。
5. 如果某些环境暂未补齐，要显式说明，不允许静默 fallback。

请直接改代码。最终回复必须包含：
- `delta_signature` 的定义、首时刻处理规则和使用方式
- visual memory update 的输入、输出和扫描时序
- encoder token 组成
- 修改文件列表
- 新增配置项
- 最小训练 / 评估 smoke test 命令
- 还未做的扩展点，例如多槽位 `num_memory_slots>1`
```

- [ ] Prompt 5：加入辅助损失与消融开关

```
你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作。当前 full SIPM-ACT 主结构应已稳定。你的任务是在不破坏原始 `recon + KL` 训练路径的前提下，新增可选辅助损失和对应消融开关，便于后续系统实验。默认情况下，这些辅助损失必须全部关闭。

你需要先详细阅读并理解`main/discussions/2026-03-18_02_signature_indexed_prefix_memory_ACT_discussion.md`的`共享实现约定`部分，这是基础。

开始前先阅读：
- `main/discussions/2026-03-18_02_signature_indexed_prefix_memory_ACT_discussion.md`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/modeling_act.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/configuration_act.py`
- `main/scripts/train_policy.py`

优先实现三项辅助损失，按这个优先级：
1. `anchor reconstruction loss`
2. `signature consistency loss`
3. `past-action prediction loss`

注意：这里把原来含糊的 “past-action / past-token” 收紧为一个更容易复现的最小版本。若后续要做 past-token 变体，可以再单独扩展。

实现要求：
1. 每项损失都必须有独立 config 开关和权重，例如：
   - `use_anchor_loss`
   - `anchor_loss_weight`
   - `use_signature_consistency_loss`
   - `signature_consistency_loss_weight`
   - `use_past_action_loss`
   - `past_action_loss_weight`
2. 默认关闭时，总训练行为必须与现有 full model 一致。
3. 缺失输入时必须自动跳过并可解释，例如：
   - 没启用 anchor 时，`anchor_loss` 自动跳过；
   - 没启用 visual prefix memory 时，所有依赖 memory 的辅助损失都自动跳过；
   - 没启用 `delta_signature` 时，不允许静默读取不存在的字段。
4. 训练日志必须分别记录：
   - `l1_loss`
   - `kld_loss`（若有）
   - `anchor_loss`
   - `signature_consistency_loss`
   - `past_action_loss`
   - `total_loss`
5. `anchor reconstruction loss` 的目标必须明确。推荐目标：
   - 用一个小头从最终 visual memory 预测 anchor token / anchor embedding；
   - target 必须来源于与主路径一致的 anchor 表示，而不是另起一套模糊目标。
6. `signature consistency loss` 必须明确默认预测目标。推荐默认：
   - 从最终 visual memory 预测当前 `g_t`；
   - 若要支持预测 `delta_signature`，必须通过显式配置切换。
7. `past_action_loss` 先做最小版本：
   - 预测上一时刻单步动作 `a_{t-1}` 或语义等价的最近已执行动作；
   - 不要为了这一步重构复杂 token-level 自监督接口。
8. 所有辅助损失的实现都要尽量低侵入，不要因为加损失而重构整套模型输入输出。
9. 提供便于 ablation 的 CLI / config 组合，支持后续做：
   - full model w/o anchor loss
   - full model w/o sig consistency
   - full model w/o past action
10. 不要引入任何需要人工事件标注的监督。

边界条件：
- 主方法核心仍然是动作块重建；辅助损失只能辅助，不能主导接口设计。
- 不要默认启用所有 loss。
- 不要破坏已有 checkpoint 的加载。

验收标准：
1. 所有辅助损失默认关闭时，行为与当前 full model 一致。
2. 单独开启任意一项损失时，训练能跑通，日志可见该项数值。
3. 对缺失输入的情况有明确保护逻辑，不会产生难懂的 shape 错误。
4. 至少提供一个最小 ablation 训练命令示例。

请直接改代码。最终回复必须包含：
- 每项辅助损失的数学目标在实现里如何落地
- 修改文件列表
- 新增配置和 CLI
- 日志中新增的字段
- 最小 ablation 命令
- 你认为哪项辅助损失最值得先做真实实验，以及原因
```

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
