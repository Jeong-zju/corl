# PRISM 固定基座双臂实验四：轨迹模板引导穿线

## 实验定位

推荐英文名：`Trajectory-Cued Bimanual Cable Weaving`

推荐数据集名：`zeno-ai/TrajectoryCuedCableWeaving`

该实验让固定基座双臂机器人根据早期可见的轨迹模板，把一根柔性线缆按指定路径绕过桌面 peg board 上的多个柱钉。模板在开始阶段可见，随后被遮挡；机器人需要在后续多个分叉点持续记住路径顺序。

该实验专门用于检验 PRISM 对连续路径信息的记忆，而不只是对离散标签或目标类别的记忆。它把 path signature 的优势放大到一个非常直观的双臂物理任务中：历史轨迹本身就是任务说明。

## PRISM 专属性

本实验突出 PRISM 的路径签名能力：

- 轨迹模板包含有顺序的空间路径，例如先绕上方柱钉、再穿中间孔、最后绕下方柱钉。
- 多个分叉点的当前视觉局部非常相似，下一步方向必须依赖早期模板和已执行路径。
- 双臂会交替执行送料、拉紧、换手和绕柱动作，delta signature 对阶段转换非常关键。
- 线缆形态会产生视觉噪声，单帧图像很难稳定反推出完整目标路径。

如果模型只靠当前图像，很容易在中后段选择错误柱钉或错误绕行方向；PRISM 应该能把早期模板和已经执行的路径编码进 slot memory。

## 固定基座双臂平台设置

机器人固定在 peg board 前方，双臂覆盖整个桌面板区域。数据格式建议：

- `observation.state`: `[0, 0, 0, left7, right7]`
- `action`: `[0, 0, 0, left7, right7]`
- 三路 RGB 输入保持与现有仓库一致。

双臂分工：

- 一只手持线缆前端，执行穿线和绕柱。
- 另一只手负责压线、拉紧、换手和防止线缆回弹。

可以在 demonstrations 中让左右手角色随机交换，避免策略只记住固定手别脚本。

## 场景布置

桌面放置一个 3x3 或 4x3 peg board。柱钉外观一致，不使用可见编号。板左侧有线缆起点，右侧有终点夹。

设计 4 条基础轨迹模板：

- Route A：上方绕行，中间直穿，下方出线。
- Route B：下方绕行，中间直穿，上方出线。
- Route C：先中间交叉，再上方绕行，再右下出线。
- Route D：先下方绕行，再中间交叉，再右上出线。

模板可以用一张起始路线卡显示，也可以在 peg board 上用短暂投影线显示 2-3 秒。随后模板遮挡或投影关闭。

关键是：后续柱钉本身不能显示路线编号，路线只能靠早期模板和已执行路径记住。

## 任务流程

### Stage 1：观察轨迹模板

机器人在初始姿态观察路线卡。路线卡显示完整路径或前几个关键转折点。

为了避免任务变成复杂文字理解，建议使用清晰曲线图或高亮路径图，而不是语言说明。

### Stage 2：固定线缆起点

一只手抓住线缆前端，另一只手将线缆根部压入起点夹。所有路线都从同一起点进入 peg board。

### Stage 3：公共起始绕线

机器人执行一段对所有路线相同的起始动作，例如从左侧起点穿入第一个中心导向环。模板此时已经不可见。

该阶段制造共享前缀，避免策略在一开始就朝不同路线分叉。

### Stage 4：多分叉穿线

机器人根据路线模板和已执行路径，在 2-4 个分叉点连续选择绕行方向。每个分叉点都需要双臂配合：

- 持线手绕过目标柱钉。
- 辅助手压住上一段线缆，保持张力。
- 必要时双臂换手，避免线缆缠绕或过度拉伸。

### Stage 5：终点固定

机器人把线缆末端压入目标终点夹，并整理线缆张力。成功后线缆应沿模板路径稳定贴合 peg board。

## 数据采集设计

建议首轮采集 240-360 条 demonstrations：

- 4 条路线，每条 60-90 条。
- 每条路线覆盖线缆初始弯曲、起点夹位置微扰、柱钉板轻微平移。
- 左手持线和右手持线都覆盖。
- 每条都包含模板消失后的公共起始绕线。

推荐分阶段采集：

- Stage A：2 条路线、2 个分叉点，验证穿线可操作性。
- Stage B：4 条路线、3 个分叉点，固定板尺度。
- Stage C：加入左右手角色随机化。
- Stage D：加入 board 尺度或位置扰动，用于测试 path signature 对轨迹形状的泛化。

## 成功标准

硬性成功：

- 模板只在开始阶段可见。
- 线缆从共同起点进入 peg board。
- 分叉点选择顺序与模板一致。
- 线缆最终固定在正确终点夹。
- 线缆没有明显脱钉、缠绕或过度拉紧。

推荐量化指标：

- 完整路线成功率。
- 分叉点选择准确率。
- 路径编辑距离：实际柱钉序列到目标柱钉序列的编辑距离。
- 终点夹成功率。
- 线缆张力异常次数或脱钉次数。
- held-out 尺度成功率：训练板尺度与评估板尺度不同。

## 对照实验与消融

建议评估：

1. Full PRISM。
2. Current Observation Only。
3. Signature Only。
4. Visual Prefix Memory Only。
5. Prefix Memory without Signature Routing。
6. No-Delta Routing。
7. Signature Depth 1/2/3。
8. Template Always Visible Upper Bound。

该实验还适合做 reference trajectory 诊断：训练时使用标准 peg board，评估时把 peg board 缩放 0.9x 或 1.1x，观察 PRISM 是否比普通 prefix memory 更稳。

## 防止捷径

需要注意：

- 路线卡关闭后不能在 peg board 上留下高亮残影。
- 线缆颜色和路线不能绑定。
- 不同路线的公共起始绕线必须一致。
- 已完成线段可以显示历史执行结果，但不能直接显示未来路线；这是允许的，因为任务本来需要根据“已执行路径 + 初始模板”判断下一步。
- 柱钉不能带可见编号或颜色。

推荐诊断：

- 只用 Stage 3 当前图像预测路线，准确率应接近 25%。
- 只用最后 1 秒状态预测完整路线，应明显低于完整前缀。
- 检查错误是否集中在路线交叉点，交叉点错误更能体现记忆/路径建模失败。

## 训练配置建议

本实验轨迹长、分叉多，建议配置比单目标任务更强：

- `prefix_train_max_steps: 32`
- `prefix_frame_stride: 4` 到 `8`
- `slot_memory_num_slots: 6`
- `signature_depth: 3`
- `use_path_signature: true`
- `use_delta_signature: true`
- `use_visual_prefix_memory: true`
- `use_signature_indexed_slot_memory: true`
- `slot_memory_use_delta_routing: true`
- `slot_memory_use_readout_pooling: true`
- `slot_memory_balance_loss_coef: 0.1`
- `slot_memory_consistency_loss_coef: 0.005`

如果穿线 episode 超过 90 秒，可以提高 `prefix_frame_stride`，保证 prefix budget 覆盖从模板读取到最终终点固定的全局历史。

## 论文展示建议

该实验适合展示：

- 分叉点准确率随分叉序号变化的曲线。
- 不同 signature depth 的路线成功率。
- held-out board scale 的泛化成功率。
- slot routing 与路线阶段对齐的可视化。

一句话叙事可以写为：机器人一开始看到一条路线，路线消失后，它要靠记忆和自己的执行轨迹把柔性线缆一步步穿过相同的柱钉阵列。
