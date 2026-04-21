# Deploy

这套部署入口是单进程方案：

- 一个 ROS1 node 直接订阅图像、双臂 joint state 和 odom
- 在 node 内部拼成一帧 observation
- 直接加载 ACT / Streaming ACT checkpoint 做 `select_action()`
- 直接发布 `Twist` 和左右臂 `JointState`

启动方式：

```bash
python3 main/deploy/ros1_adapter/ros1_adapter_node.py \
  --config main/deploy/configs/deploy_zeno_act.yaml
```

主要可调项都在 YAML 里：

- `policy.path`：checkpoint 路径，按当前 YAML 文件的相对路径解析；也可以直接写训练输出根目录，deploy 会自动解析最新 run
- `policy.device`：推理设备
- `ros.topics.*`：订阅/发布话题
- `runtime.control_hz`：推理控制频率
- `ros.joint_names_left.name` / `ros.joint_names_right.name`
- `image.width` / `image.height` / `image.color_order`
- `command.max_linear_x` / `max_linear_y` / `max_angular_z`
- `debug.enabled` / `debug.publish_hz`：开启实机视觉调试图发布

说明：

- 现在默认会按 `[base_vx, base_vy, base_omega, left7, right7]` 组 17 维状态。
- 不做时延/陈旧检测，只使用每个 topic 当前最新的一条消息。
- 如果你的 checkpoint 依赖 `path_signature` / `delta_signature`，node 现在会优先按 checkpoint 真实配置自动开启在线 signature 计算，不再要求你在 deploy YAML 里手工保持一致。
- 如果 checkpoint 把 signature 特征标记成 `pre_normalized_observation_keys`，deploy 会自动从训练 run 关联的数据集 `meta/stats.json` 读取统计量，对在线 `path_signature` / `delta_signature` 做同样的归一化；为了抑制真实机器人噪声在零方差维度上的放大，归一化前还会先裁到训练数据的 `min/max` 支持范围内。
- 对 `streaming_act` 的 PRISM 变体，在线部署走的是“当前帧 + 在线 visual prefix memory / slot memory 更新”路径，不需要每步重建显式 prefix sequence tensor。
- 如果图像颜色和训练时不一致，优先改 YAML 里的 `image.color_order`。

视觉调试：

打开 YAML 里的 `debug.enabled: true` 后，deploy 会额外发布三路 `sensor_msgs/Image`：

```text
/deploy/debug/attention
/deploy/debug/slot_memory
/deploy/debug/signature
```

可以用 `rqt_image_view` 分别查看。`attention` 是三路当前视觉输入叠加 decoder cross-attention heatmap；`slot_memory` 显示 SISM routing、zero-signature routing baseline、signature 引起的 routing delta、write gate / readout / memory delta；`signature` 显示当前 path signature 与数据集统计的对比。如果数据集 `meta/stats.json` 里没有 `observation.path_signature`，该面板会提示缺少数据集 signature 分布统计。
