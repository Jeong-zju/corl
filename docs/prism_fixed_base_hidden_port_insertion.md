# PRISM 固定基座双臂实验一：隐式端口记忆插接

## 实验定位

推荐英文名：`Hidden-Cue Bimanual Port Insertion`

推荐数据集名：`zeno-ai/HiddenCueBimanualPortInsertion`

该实验面向固定基座双臂机器人，核心任务是：机器人在任务开始时短暂看到一个目标端口提示，随后提示消失；机器人需要完成双臂取插头、开盖、整理线缆、共享中间姿态，最后仍然根据早期提示把插头插入正确端口。

该实验专门用于检验 PRISM 的早期视觉记忆、signature-indexed slot memory 写入/读出，以及长延迟下的目标保持能力。它不是普通端到端插接任务，因为最终阶段的当前画面被刻意设计为对四个端口几乎等价，正确端口只能由早期隐藏提示和执行前缀恢复。

## PRISM 专属性

本实验突出 PRISM 的三个机制。

1. 早期提示只出现一次，后续不可见，要求 visual prefix memory 写入初始证据。
2. 取插头、开盖、收线、移动到中立姿态是一段所有条件共享的动作序列，要求 slot memory 在长公共前缀后仍能读出目标端口。
3. 四个端口的空间位置相近、外观相同，最终动作差异主要体现在双臂末端微小分支，因此需要 path signature 和 delta signature 帮助区分执行前缀阶段。

对普通 ACT、Diffusion Policy 或当前观测策略而言，最终时刻只有一个对称端口板和手中插头，缺少早期提示；如果模型仍然成功，通常说明环境泄漏了捷径。因此本实验可以作为 PRISM 记忆能力的高信号物理实验。

## 固定基座双臂平台设置

机器人固定在桌前，不发布底盘运动命令。为兼容当前代码仓库的数据接口，数据仍可使用 17 维格式：

- `observation.state`: `[0, 0, 0, left7, right7]`
- `action`: `[0, 0, 0, left7, right7]`
- 图像 key: `observation.images.realsense_top`、`observation.images.realsense_left`、`observation.images.realsense_right`

部署时建议设置固定基座约束：

- `command.publish_base: false`，或保持 `cmd_vel` 三维动作恒为 0。
- path signature 仍从完整 `observation.state` 计算，前三个零维不会影响双臂轨迹的历史编码。
- 控制频率建议 20-30 Hz，图像分辨率 224x224，与现有 Zeno 数据转换配置保持一致。

## 场景布置

桌面中央放置一个 2x2 端口板，四个端口外观完全一致，编号只用于实验记录，不对机器人可见。

建议端口布局：

- 左上：Port A
- 右上：Port B
- 左下：Port C
- 右下：Port D

桌面左侧放置插头和柔性线缆，右侧放置端口保护盖或滑盖。机器人双臂分工如下：

- 右臂负责抓取插头、执行主要插接动作。
- 左臂负责打开端口盖、压住线缆、辅助整理线缆张力。

初始提示可以采用下列任一种形式：

- 目标端口旁短暂亮灯 2 秒，随后熄灭。
- 顶部小屏幕显示目标端口图案 2 秒，随后转为统一空白屏。
- 端口盖外侧出现一个可移除目标标记，机器人开始动作后被机械遮挡。

关键要求是：到最终插接阶段，四个端口的当前视觉不能暴露哪一个是目标端口。

## 任务流程

每条 episode 分为五个阶段。

### Stage 1：观察隐藏提示

机器人停在固定初始姿态，三个相机能看到端口板和目标提示。目标提示持续 2-3 秒后消失。

本阶段只允许机器人通过视觉看到目标端口，不提供任务 id、端口编号、语言标签或其他特权变量。

### Stage 2：双臂准备

右臂从插头座中取出插头；左臂打开端口保护盖或移动遮挡片。该阶段对所有目标端口尽量相同，避免通过动作早期差异直接编码端口。

建议加入轻微扰动：

- 插头初始角度在小范围内随机。
- 线缆弯曲形态随机。
- 端口盖摩擦或开合角度轻微变化。

### Stage 3：公共中立段

双臂把插头和线缆整理到同一个中立姿态。例如右臂将插头移动到端口板正前方中央，左臂把线缆压到固定限位槽内。

这一段是实验的记忆延迟核心。四个目标端口都必须经过几乎相同的中立段，不能提前朝目标端口移动。

可以做延迟 sweep：

- 0 秒：直接从准备阶段进入插接。
- 3 秒：在中立姿态保持并做轻微线缆整理。
- 6 秒：增加一次双臂同步 reposition。
- 10 秒：加入长时间线缆张力整理或等待动作。

### Stage 4：目标端口选择

机器人根据早期提示选择正确端口。左臂固定线缆或端口板边缘，右臂将插头移动到目标端口前方。

最终选择阶段不允许端口提示重新出现。四个端口的视觉外观保持一致。

### Stage 5：插接与保持

右臂完成插接，左臂释放线缆张力。插头插入后保持 1 秒。若使用力/位移传感器，可记录插入力峰值和插入深度；若只用视觉评估，可用端口内插头深度和位置误差判断成功。

## 数据采集设计

建议首轮采集 160-240 条 demonstrations：

- 4 个目标端口，每个端口 40-60 条。
- 每个端口覆盖不同插头初始角度、线缆弯曲和端口盖初始角度。
- 每条 episode 都包含完整提示消失后的公共中立段。
- 采集时保持任务节奏一致，不要让某个端口对应特殊停顿或特殊手势。

推荐分阶段采集：

- Stage A：只做两个端口，验证插接动作和端口遮挡设计。
- Stage B：扩展到四个端口，加入提示隐藏。
- Stage C：加入 3/6/10 秒公共中立延迟。
- Stage D：加入端口板轻微平移、插头角度扰动和线缆扰动。

## 成功标准

硬性成功条件：

- 目标提示只在任务开始阶段可见。
- 机器人完成双臂取插头、开盖和线缆整理。
- 最终选择的端口与初始提示一致。
- 插头插入正确端口并保持稳定。
- 没有人工中途提示或重置。

推荐量化指标：

- 端口选择准确率。
- 插接成功率。
- 错端口插接率。
- 插接前定位误差。
- 插入力峰值或插入尝试次数。
- 任务阶段进度：提示读取、准备完成、中立段完成、正确端口选择、插接成功。

## 对照实验与消融

建议至少做六组评估：

1. Full PRISM：开启 path signature、delta signature、visual prefix memory、signature-indexed slot memory。
2. Current Observation Only：关闭 PRISM 记忆，只用当前观测。
3. Signature Only：只保留 path/delta signature，不使用 visual prefix memory。
4. Prefix Memory without Signature Routing：保留 prefix memory，但关闭 signature-indexed slot routing。
5. No-Delta Routing：保留 signature routing，但关闭 delta routing。
6. Shuffled Cue：评估时打乱初始提示与真实目标端口映射，检查策略是否系统性跟随错误提示。

该实验特别推荐展示 slot routing 可视化：在 Stage 1 写入提示，在 Stage 3 公共中立段保持，在 Stage 4 读出目标端口。

## 防止捷径

需要重点排查下列泄漏：

- 端口提示熄灭后仍有余光、反光或小屏残影。
- 四个目标端口对应的公共中立段动作不一致。
- 端口板周围线缆形态因目标端口不同而提前分化。
- 操作者在采集时对不同端口有不同等待节奏。
- 空间标号、贴纸或环境纹理让某个端口更容易被识别。

推荐做当前观测 probe：截取 Stage 4 插接前图像，训练一个小分类器预测目标端口。如果分类器显著高于随机水平，说明当前观测泄漏了目标信息，需要重新遮挡或统一场景。

## 训练配置建议

建议沿用当前 PRISM Diffusion 或 Streaming ACT 的记忆设置：

- `use_path_signature: true`
- `use_delta_signature: true`
- `use_prefix_sequence_training: true`
- `prefix_train_max_steps: 24`
- `prefix_frame_stride: 4` 到 `8`
- `use_visual_prefix_memory: true`
- `use_signature_indexed_slot_memory: true`
- `slot_memory_num_slots: 4`
- `slot_memory_use_delta_routing: true`
- `slot_memory_use_readout_pooling: true`
- `slot_memory_balance_loss_coef: 0.05` 到 `0.1`
- `slot_memory_consistency_loss_coef: 0.001` 到 `0.005`

如果使用 PRISM Diffusion，固定基座插接建议把在线执行设为更短动作队列：

- `n_action_steps: 1` 到 `10`
- 对长 horizon 训练可保留 `horizon: 64` 或 `104`

原因是插接末端需要及时响应当前接触和视觉误差，过长 action queue 会让新更新的 PRISM memory 不能立刻影响已排队动作。

## 论文展示建议

该实验适合展示三类结果：

- 主表：端口选择准确率和完整插接成功率。
- 延迟图：提示消失到端口选择之间的 0/3/6/10 秒延迟 sweep。
- 可视化：三帧 rollout 图像配 slot routing 热图，显示提示写入、中立段保持、最终读出。

一句话叙事可以写为：机器人只在开始时看到目标端口，随后经历一段与目标无关的双臂准备动作，最后仍然记得该把插头插到哪里。
