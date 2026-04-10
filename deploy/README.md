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

- `policy.path`：checkpoint 路径，按当前 YAML 文件的相对路径解析
- `policy.device`：推理设备
- `ros.topics.*`：订阅/发布话题
- `runtime.control_hz`：推理控制频率
- `ros.joint_names_left.name` / `ros.joint_names_right.name`
- `image.width` / `image.height` / `image.color_order`
- `command.max_linear_x` / `max_linear_y` / `max_angular_z`

说明：

- 现在默认会按 `[base_vx, base_vy, base_omega, left7, right7]` 组 17 维状态。
- 不做时延/陈旧检测，只使用每个 topic 当前最新的一条消息。
- 如果你的 checkpoint 依赖 `path_signature` / `delta_signature`，node 会在线计算。
- 如果图像颜色和训练时不一致，优先改 YAML 里的 `image.color_order`。
