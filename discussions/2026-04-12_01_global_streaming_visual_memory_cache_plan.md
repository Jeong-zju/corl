# 使用冻结前缀编码器的全局流式视觉记忆

## 目标

我们希望 visual prefix memory 成为真正的流式记忆：

```text
M_t = U(M_{t-1}, E_prefix(o_t), s_t, g_t, delta_t)
```

记忆状态 `M_t` 具有固定大小，但它的更新历史覆盖完整 episode 前缀 `1:t`。这不同于当前的稀疏前缀重建路径；当前训练只扫描由 `prefix_train_max_steps` 和 `prefix_frame_stride` 选出的有界子集。

实际挑战是：在 `wholebody_hanger_stage3_v30` 上用原始 RGB 帧做完整前缀训练代价太高。该数据集有 3 个相机、224x224 视频，episode 最长约 3530 帧。直接扫描完整前缀图像会反复解码和编码过多帧。

建议方案是：

```text
training:
  video frame -> frozen E_prefix -> cached z_t
  cached z_1:t -> full streaming memory scan -> M_t -> action expert loss

inference:
  live frame -> same frozen E_prefix -> z_t
  M_{t-1}, z_t -> online memory update -> M_t -> action expert prediction
```

缓存只是加速手段。模型定义中仍然包含推理时使用的同一个冻结前缀视觉编码器。

## 理论表述

不要宣称该方法保证在每次运行中都能经验性地优于 ACT。仅凭架构无法保证这一点。

更稳妥的主张是：

1. 策略除了 action expert 的当前观测外，还接收一个确定性的额外历史摘要。
2. 记忆状态具有固定预算，但会在完整视觉前缀上更新。
3. action expert 路径可以通过忽略 memory condition 来恢复，例如使用零初始化 residual adapters，或通过配置禁用 memory。
4. 因此，从原则上说，该模型族的表达能力不弱于对应的 action expert 模型族，同时能访问严格更多的历史信息。

建议论文措辞：

> The memory-augmented policy contains the base action expert as a degenerate case in which the memory adapter is ignored, while additionally exposing a fixed-budget summary of the complete execution prefix. This gives the model access to information unavailable to a current-observation-only expert, without increasing memory size with episode length.

## 当前实现缺口

推理已经在线更新 visual prefix memory。在 `StreamingACTPolicy.select_action` 中，策略会在每个控制步调用在线记忆更新，并保存 `_visual_prefix_memory_state`。

训练目前会从 `select_prefix_positions` 构造出的前缀张量重建记忆。该选择器：

1. 按 `prefix_frame_stride` 采样位置。
2. 始终保留当前步。
3. 如果采样前缀超过 `prefix_train_max_steps`，则保留第一个位置和最近的尾部位置。

因此当前训练路径是：

```text
training:
  sparse selected prefix images -> memory scan -> M_t

inference:
  every online step image -> memory update -> M_t
```

当我们希望宣称使用完整视觉历史记忆时，这里存在语义不匹配。

## 建议的数据契约

新增一种训练模式：

```yaml
visual_prefix_memory_training_mode: "full_prefix_cached"
streaming_prefix_max_steps: 0  # 0 表示自动解析为最大 episode 长度
```

保留当前稀疏模式作为消融：

```yaml
visual_prefix_memory_training_mode: "sparse_images"
prefix_train_max_steps: 16
prefix_frame_stride: 5
```

在 `full_prefix_cached` 模式下，时间 `t` 的每个训练样本应提供：

```text
当前 action expert 输入：
  observation.images.*                     (C, H, W)，每个相机的当前帧
  observation.state                        (state_dim,)
  observation.path_signature               (signature_dim,)
  observation.delta_signature              (signature_dim,)
  action                                   (chunk_size, action_dim)
  action_is_pad                            (chunk_size,)

完整前缀记忆输入：
  observation.prefix_visual_memory_embedding     (T_mem, D_mem)
  observation.prefix_state                       (T_mem, state_dim)
  observation.prefix_path_signature              (T_mem, signature_dim)
  observation.prefix_delta_signature             (T_mem, signature_dim)
  observation.prefix_mask                        (T_mem,)
```

`T_mem` 是 `streaming_prefix_max_steps`；默认情况下它等于最大 episode 长度。`prefix_mask[:t+1] = True`，其余位置右侧 padding。`prefix_visual_memory_embedding[i]` 必须等于用推理时完全相同的冻结编码器和图像预处理计算出的 `E_prefix(o_i)`。

如果深度为 3 的 signatures 对完整前缀张量来说过重，可以使用以下变体之一：

1. 用 `signature_depth=2` 重新计算数据集，将 17 维状态对应的 signature 维度从 5219 降到约 306。
2. 缓存冻结的投影 signature embeddings，但仅当 signature projection 被明确定义为冻结 tokenizer 的一部分时才这样做。不要缓存可训练 projection 的输出。

建议 deadline 版本：

```text
full-prefix cached visual embeddings
+ depth-2 path/delta signatures
+ trainable signature projection
```

## 冻结前缀视觉编码器

定义一个单独的冻结前缀视觉编码器，不使用可训练的 ACT 图像 backbone：

```text
E_prefix(o_t) = pooled ResNet18 layer4 feature, shape (512,)
```

推荐来源：

1. 从已经在 `wholebody_hanger_stage3_v30` 上训练过的 ACT 或 Streaming ACT checkpoint 中加载视觉 backbone。
2. 冻结所有 `E_prefix` 参数。
3. 缓存生成和在线推理使用同一个模块。

备选来源：

1. 使用 ImageNet `ResNet18_Weights.IMAGENET1K_V1`。
2. 冻结它。
3. 在缓存元数据和实验日志中清楚记录该来源。

重要一致性规则：

1. 缓存原始冻结编码器输出，例如 pooled `(512,)` features。
2. 如果模型需要一个可训练 projection 到 `dim_model`，则在训练时加载 cache 之后应用该 projection，并在推理时在线计算 `E_prefix` 之后应用同一个 projection。
3. 不要缓存可训练层的输出。
4. 在 `cache_meta.json` 中记录图像 normalization stats、camera list、checkpoint path/hash、dtype、feature dimension、pooling type 和 code version。

## 缓存布局

推荐根目录：

```text
data/zeno-ai/wholebody_hanger_stage3_v30/.cache/prefix_visual_embeddings/<fingerprint>/
```

推荐文件：

```text
cache_meta.json
embeddings.fp16.memmap
episode_index.json
```

如果在写入前对相机做平均，`embeddings.fp16.memmap` 可以存储一个 flat array：

```text
(num_frames, D_mem)
```

如果保留 per-camera memory inputs：

```text
(num_frames, num_cameras, D_mem)
```

为了 deadline 效率，在缓存时平均相机 embeddings：

```text
z_t = mean_camera(E_prefix(o_t^camera))
```

这与当前 visual prefix memory 实现一致：当前实现会在记忆更新前平均 camera embeddings。

## 训练语义

对于属于 episode `[ep_start, ep_end)` 的样本索引 `idx`，构造：

```text
prefix_abs_indices = [ep_start, ep_start + 1, ..., idx]
```

然后收集缓存 embeddings 和低维特征：

```text
prefix_visual_memory_embedding = cache[prefix_abs_indices]
prefix_state = hf_dataset["observation.state"][prefix_abs_indices]
prefix_path_signature = hf_dataset["observation.path_signature"][prefix_abs_indices]
prefix_delta_signature = hf_dataset["observation.delta_signature"][prefix_abs_indices]
prefix_mask = True for valid positions
```

把所有 prefix tensors 右侧 padding 到 `T_mem`。

模型随后执行：

```text
hidden = zeros(batch, num_slots, dim_model)
for i in range(T_mem):
    if prefix_mask[:, i]:
        hidden = U(hidden, z_i, state_i, signature_i, delta_i)
M_t = hidden
```

这就是完整前缀流式重建。不会使用未来帧。

## 推理语义

在 episode reset 时：

```text
policy.reset()
M = None
```

在控制步 `t`：

```text
z_t = E_prefix(current_image_t)
M_t = U(M_{t-1}, z_t, state_t, signature_t, delta_t)
action_chunk = action_expert(current_observation_t, M_t)
```

除了训练从 cache 读取 `z_i` 而推理在线计算 `z_t` 外，两者语义相同。

## 配置草案

建议新增：

```yaml
train:
  visual_prefix_memory_training_mode: "full_prefix_cached"
  streaming_prefix_max_steps: 0
  prefix_visual_embedding_cache_root: ".cache/prefix_visual_embeddings"
  prefix_visual_encoder_source: "checkpoint"
  prefix_visual_encoder_checkpoint_path: "outputs/train/zeno-ai/wholebody_hanger_stage3_v30/act/<run>/checkpoints/<step>/pretrained_model"
  prefix_visual_encoder_freeze: true
  prefix_visual_embedding_dim: 512
  prefix_visual_embedding_dtype: "float16"
```

为稀疏消融保留旧字段：

```yaml
train:
  prefix_train_max_steps: 16
  prefix_frame_stride: 5
```

规则：

```text
if visual_prefix_memory_training_mode == "full_prefix_cached":
    ignore prefix_train_max_steps and prefix_frame_stride for visual memory
elif visual_prefix_memory_training_mode == "sparse_images":
    use prefix_train_max_steps and prefix_frame_stride
```

## Coding Prompts（编码提示）

通用要求：

1. 每个实现任务开始前，先阅读并理解 `main/discussions/2026-04-12_01_global_streaming_visual_memory_cache_plan.md`。
2. 你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作；需要运行 Python、训练、测试或脚本时使用该环境。

### Prompt 1：新增冻结前缀视觉编码器模块

```text
你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作。暂时不要修改训练 defaults。

目标：
为 Streaming ACT 新增一个冻结前缀视觉编码器抽象。它必须同时可用于离线 cache builder 和在线推理。

先阅读：
- `main/discussions/2026-04-12_01_global_streaming_visual_memory_cache_plan.md`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/modeling_streaming_act.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/configuration_streaming_act.py`
- `main/scripts/train_policy.py`
- `main/scripts/eval_policy.py`
- `main/scripts/eval_helpers.py`

要求：
1. 添加 config fields，默认禁用：
   - `use_frozen_prefix_visual_encoder: bool = False`
   - `prefix_visual_encoder_source: str = "imagenet"`，可选值为 `"imagenet"` 和 `"checkpoint"`
   - `prefix_visual_encoder_checkpoint_path: str | None = None`
   - `prefix_visual_embedding_dim: int = 512`
   - `prefix_visual_encoder_camera_pool: str = "mean"`
2. 实现一个 module，接收当前已 normalized 的 camera images，并返回 `(B, D_mem)`。
3. 如果 source 是 `"imagenet"`，从 `ResNet18_Weights.IMAGENET1K_V1` 初始化。
4. 如果 source 是 `"checkpoint"`，从现有 policy checkpoint 加载 prefix encoder backbone weights。如果 checkpoint 格式有歧义，要用清晰错误信息直接失败。
5. 冻结全部 prefix visual encoder 参数，并将其设为 eval mode。
6. 不要替换 action expert 的当前图像 backbone。prefix visual encoder 是独立的。
7. 添加一个小方法，例如 `encode_prefix_visual_memory_observation(batch)`，并且只在明确请求的位置使用它。
8. 添加 smoke test 或最小脚本路径，用于构造 encoder 并检查输出 shape。

一致性约束：
1. Cache generation 和 online inference 必须使用这个完全相同的 encoder 定义。
2. 不要缓存或暴露可训练层的输出作为冻结视觉特征。
3. 当 `use_frozen_prefix_visual_encoder=False` 时，当前行为必须保持不变。

最终回复必须包括：
- 修改的文件
- 新增 config fields
- 精确 tensor 输入/输出 shapes
- Smoke test command
```

### Prompt 2：构建前缀视觉 embedding cache

```text
你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作。为完整前缀视觉记忆实现离线 cache generation。暂时不要修改模型训练路径。

先阅读：
- `main/discussions/2026-04-12_01_global_streaming_visual_memory_cache_plan.md`
- `main/scripts/train_policy.py`
- `main/data/process_dataset.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/modeling_streaming_act.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/prefix_sequence.py`

目标：
创建一个脚本或数据集处理操作，为每个数据集 frame 计算 `E_prefix(o_t)`，并将结果存储为可复用 cache。

要求：
1. 添加脚本，例如：
   `main/scripts/cache_prefix_visual_embeddings.py`
2. Inputs：
   - `--dataset-root`
   - `--dataset-repo-id`
   - `--output-root`
   - `--encoder-source imagenet|checkpoint`
   - `--encoder-checkpoint-path`
   - `--camera-keys` 可选，默认所有常规 observation cameras
   - `--camera-pool mean`
   - `--dtype float16|float32`
   - `--batch-size`
   - `--num-workers`
   - `--video-backend`
   - `--use-imagenet-stats` 或等价的 normalization selection
3. 用训练时相同的视觉 normalization policy 加载数据集。
4. 每帧只解码一次，运行冻结前缀视觉编码器，并存储 memmap：
   - averaged cameras: `(num_frames, D_mem)`
   - 或 per-camera: `(num_frames, num_cameras, D_mem)`
5. 写入 `cache_meta.json`，包含：
   - dataset root
   - dataset metadata fingerprint
   - number of frames
   - episode spans
   - camera keys
   - encoder source
   - checkpoint path 和 sha256（如果使用 checkpoint）
   - image normalization mode 和 stats fingerprint
   - dtype
   - embedding shape
   - code version（如果可用）
6. 启动时，如果已有 cache 的 metadata 匹配，则复用它，除非传入 `--force`。
7. 添加 verification mode：随机采样 frame indices，检查 shapes 和 finite values。

不要：
1. 不要存储完整 RGB prefix tensors。
2. 不要缓存可训练 model projections 的输出。
3. 如果 normalization metadata 或 checkpoint hash 不匹配，不要静默继续。

最终回复必须包括：
- 修改的文件
- Cache directory layout
- `wholebody_hanger_stage3_v30` 的示例 command
- Verification command
```

### Prompt 3：添加完整前缀 cached dataset wrapper

```text
你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作。添加一个 dataset wrapper，将完整前缀 cached visual embeddings 送入 Streaming ACT training。不要移除现有 sparse prefix image path。

先阅读：
- `main/discussions/2026-04-12_01_global_streaming_visual_memory_cache_plan.md`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/prefix_sequence.py`
- `main/scripts/train_policy.py`

目标：
支持 `visual_prefix_memory_training_mode="full_prefix_cached"`，并提供完整 episode-prefix memory inputs。

要求：
1. 添加 config field：
   - `visual_prefix_memory_training_mode: str = "sparse_images"`
   - valid values: `"sparse_images"`, `"full_prefix_cached"`
   - `streaming_prefix_max_steps: int = 0`，其中 0 表示从 metadata 取最大 episode length
   - `prefix_visual_embedding_cache_dir: str | None = None`
2. 在 `"sparse_images"` mode 中，当前行为必须完全保持不变。
3. 在 `"full_prefix_cached"` mode 中，dataset samples 应返回：
   - `observation.prefix_visual_memory_embedding`: `(T_mem, D_mem)`
   - `observation.prefix_state`: `(T_mem, state_dim)`
   - `observation.prefix_path_signature`: `(T_mem, signature_dim)`，如果启用
   - `observation.prefix_delta_signature`: `(T_mem, signature_dim)`，如果启用
   - `observation.prefix_mask`: `(T_mem,)`
4. Prefix ordering 必须从旧到新，最后一个 valid element 必须是当前 frame。
5. 当 `streaming_prefix_max_steps=0` 时，valid prefix 应包含从 episode start 到当前 index 的每一帧。
6. 如果 `streaming_prefix_max_steps` 为正且短于当前 prefix，默认直接报错。仅在必要时添加一个显式 future option，例如 `full_prefix_truncation_policy`，但不要静默截断。
7. 在训练开始前验证 cache metadata 与 dataset metadata。
8. 添加 feature overrides，使 policy 将 `observation.prefix_visual_memory_embedding` 视为 state-like tensor 或新的受支持 feature type。除非显式配置，否则 cached embeddings 使用 identity normalization。
9. 确保当前 observation images 仍然存在，供 action expert 使用。

重要：
对于 full global visual memory，`prefix_train_max_steps` 和 `prefix_frame_stride` 不得控制 visual memory prefix。它们只保留给 `"sparse_images"` 消融。

最终回复必须包括：
- 修改的文件
- 新 dataset mode semantics
- Tensor shapes
- 最小 dataset item smoke test
```

### Prompt 4：为 cached 完整前缀记忆添加模型 forward path

```text
你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作。实现 Streaming ACT 模型路径，使其扫描 cached full-prefix visual embeddings。

先阅读：
- `main/discussions/2026-04-12_01_global_streaming_visual_memory_cache_plan.md`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/modeling_streaming_act.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/configuration_streaming_act.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/prefix_sequence.py`

目标：
当 `visual_prefix_memory_training_mode="full_prefix_cached"` 时，通过扫描 cached embeddings 构建 visual memory，而不是扫描 prefix images。

要求：
1. 添加常量：
   - `PREFIX_VISUAL_MEMORY_EMBEDDING_KEY = "observation.prefix_visual_memory_embedding"`
2. 添加 shape `(B, T_mem, D_mem)` 验证。
3. 添加一个 trainable projection：
   - 如果 `D_mem != dim_model`，使用 `cached_visual_memory_input_proj: nn.Linear(D_mem, dim_model)`；如果相等，则使用 identity。
   该 projection 同时应用于 cached training embeddings 和 online frozen encoder outputs。
4. 添加方法：
   - `_compute_visual_prefix_memory_token_from_cached_prefix(batch)`
5. 在 projecting cached visual embeddings 后复用 `_scan_visual_prefix_memory(...)`。
6. 保持现有 `_compute_visual_prefix_memory_token_from_prefix_sequence(...)` 不变，用于 `"sparse_images"`。
7. Online inference 必须使用同一个冻结前缀视觉编码器和同一个 `cached_visual_memory_input_proj`，然后再调用 `_update_visual_prefix_memory_step`。
8. 如果启用了 cached mode，但 inference 缺少 frozen prefix visual encoder config，要用清晰错误信息直接失败。
9. Sparse mode 下当前行为必须保持不变。

保持理论表述的 adapter 要求：
1. 确保 action expert 可以忽略 memory-conditioned path。对于任何新的 conditioning path，优先使用零/identity 初始化的 residual 或 FiLM gate。
2. 不要在初始化时强迫模型依赖 memory。
3. 记录 base action expert 可以通过忽略 memory condition 恢复。

最终回复必须包括：
- 修改的文件
- 新 forward path
- 精确 training/inference consistency statement
- 单个 forward batch 的 smoke test command
```

### Prompt 5：接入训练 CLI 和 defaults

```text
你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作。将新的 full-prefix cached memory mode 接入 training defaults 和 CLI，但不要删除 sparse prefix settings。

先阅读：
- `main/discussions/2026-04-12_01_global_streaming_visual_memory_cache_plan.md`
- `main/scripts/train_policy.py`
- `main/bash/defaults/zeno-ai/wholebody_hanger_stage3_v30/streaming_act.yaml`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/configuration_streaming_act.py`

要求：
1. 添加 CLI flags：
   - `--visual-prefix-memory-training-mode`
   - `--streaming-prefix-max-steps`
   - `--prefix-visual-embedding-cache-dir`
   - `--use-frozen-prefix-visual-encoder`
   - `--prefix-visual-encoder-source`
   - `--prefix-visual-encoder-checkpoint-path`
   - `--prefix-visual-embedding-dim`
2. 在 training startup summary 中打印这些 fields。
3. 验证：
   - cached mode 要求 `use_visual_prefix_memory=True`
   - cached mode 要求 cache dir
   - cached mode 要求 frozen prefix visual encoder config，以保证 inference consistency
   - sparse mode 继续要求 prefix images
4. 只有 smoke tests 通过后，才更新 `wholebody_hanger_stage3_v30/streaming_act.yaml`。第一版提交配置使用保守设置：
   - `visual_prefix_memory_training_mode: "full_prefix_cached"`
   - `streaming_prefix_max_steps: 0`
   - `prefix_visual_embedding_dim: 512`
   - 保留 `prefix_train_max_steps` 和 `prefix_frame_stride`，但 annotate 或 log 说明它们在 cached mode 中被忽略
5. 不要修改 ACT 或 diffusion defaults。

最终回复必须包括：
- 修改的文件
- CLI examples
- Validation behavior
- Defaults 是否被修改
```

### Prompt 6：评估和部署一致性

```text
你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作。确保 evaluation 和 deployment 使用与 cached training 相同的冻结前缀视觉编码器。

先阅读：
- `main/discussions/2026-04-12_01_global_streaming_visual_memory_cache_plan.md`
- `main/scripts/eval_policy.py`
- `main/scripts/eval_helpers.py`
- `main/deploy/policy_runtime/loader.py`
- `main/deploy/policy_runtime/preprocess.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/modeling_streaming_act.py`

要求：
1. 当 trained policy config 表示使用了 cached full-prefix memory 时，inference 必须从记录的 config/checkpoint 初始化冻结前缀视觉编码器。
2. Reset 时必须清空 visual memory state。
3. 每一步必须用 `E_prefix` 编码当前 normalized images，经由同一个 cached visual memory projection 投影，并写入 memory。
4. 添加 debug stats，报告：
   - frozen prefix visual encoder 是否启用
   - encoder source
   - memory update count
   - memory state norm
   - training 时使用的 cache mode
5. 如果 deployment 无法加载 prefix visual encoder checkpoint，要直接失败。

最终回复必须包括：
- 修改的文件
- Eval/deploy consistency explanation
- Smoke test 或 dry-run command
```

### Prompt 7：Benchmark matrix 和 acceptance tests

```text
你在 `/home/jeong/zeno/corl` 仓库和 conda 环境 `corl-py312` 中工作。为全局流式 cached visual memory 添加轻量 benchmark commands 和 acceptance checks。

先阅读：
- `main/discussions/2026-04-12_01_global_streaming_visual_memory_cache_plan.md`

目标：
确认新模式比 sparse image prefix scanning 更快，并且在语义上更接近 online full-prefix memory。

Benchmark matrix：
1. ACT baseline
2. 当前 Streaming ACT sparse prefix images
3. full-prefix cached visual memory + depth-3 signatures
4. full-prefix cached visual memory + depth-2 signatures
5. full-prefix cached visual memory with memory adapter disabled or gated to zero

Metrics：
- `train/update_s`
- `train/dataloading_s`
- GPU max memory
- first 100/1000 step wall time
- loss sanity

Acceptance tests：
1. Cached train forward 对 tiny synthetic episode 产生的 memory update sequence shape，应与 online recurrent update 一致。
2. 最后一个 valid prefix element 是被监督的当前 step。
3. 不允许未来帧进入 prefix。
4. Cache metadata mismatch 必须在训练前失败。
5. 当 memory adapter disabled 或 gate 为 zero 时，行为应在预期 numerical tolerance 内等价于 base action expert path。

最终回复必须包括：
- Benchmark commands
- Acceptance test commands
- 观察到的任何 bottlenecks
```

## 推荐执行顺序

1. Prompt 1：冻结前缀视觉编码器。
2. Prompt 2：视觉 embedding cache generation。
3. Prompt 3：完整前缀 cached dataset wrapper。
4. Prompt 4：cached model scan path。
5. Prompt 5：training CLI/default wiring。
6. Prompt 6：eval/deploy consistency。
7. Prompt 7：benchmark 和 acceptance tests。

不要跳过 Prompt 6。否则 cached training 无法保证 inference consistency。
