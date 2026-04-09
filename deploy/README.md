# Deploy Bridge

`main/deploy/` 提供了一个三层部署骨架：

- `policy_runtime/`: 常驻加载 `main/` 训练出的 `act` / `streaming_act` checkpoint
- `bridge/`: 负责同步 ROS1 传感器流、在线计算 signature、请求 policy、拆解动作
- `ros1_adapter/`: 订阅 ROS1 topic 并发布标准 ROS1 控制消息

## 依赖

- `policy_runtime/`: `torch`, `lerobot`, `pyzmq`
- `bridge/`: `numpy`, `pyzmq`, 可选 `signatory`
- `ros1_adapter/`: `rospy`, `sensor_msgs`, `nav_msgs`, `trajectory_msgs`, `geometry_msgs`, `opencv-python`, `pyzmq`
- 如果配置文件使用 YAML，还需要 `PyYAML`

## 启动顺序

1. 启动 policy runtime
```bash
python3 main/deploy/policy_runtime/server.py \
  --policy-type streaming_act \
  --policy-path main/outputs/train/.../checkpoints/last/pretrained_model \
  --device cuda \
  --bind tcp://*:5555 \
  --control-bind tcp://*:5558
```

2. 启动 bridge core
```bash
python3 main/deploy/bridge/bridge_core.py \
  --config main/deploy/configs/deploy_zeno_streaming_act.yaml
```

3. 在 ROS1 机器上启动 adapter
```bash
python3 main/deploy/ros1_adapter/ros1_adapter_node.py \
  --config main/deploy/configs/deploy_zeno_streaming_act.yaml
```

## 当前默认动作映射

- `action[0:3]`: 底盘，默认关闭下发
- `action[3:10]`: 左臂 7 维目标
- `action[10:17]`: 右臂 7 维目标

对于当前 zeno 配置，这两个 7 维目标的语义是：

- `joint1 .. joint6`
- `gripper`

也就是说，bridge 当前会把每只手臂的最后 1 维和前 6 个关节一起按 `JointTrajectory` 下发。前提是你的实际控制器也接受这种 6 DoF + gripper 的 7 维联合轨迹。

## Signature Backend

对于 `streaming_act` 的 zeno checkpoint，默认应使用：

```yaml
signature_backend: auto
```

不要用：

```yaml
signature_backend: simple
```

原因是当前数据集的 `observation.path_signature` 维度是 `5219`，而 `simple` backend 只会产生 `state_dim * depth` 的近似特征。对 zeno 的 17 维状态和 3 阶 depth，这只会得到 `51` 维，无法和 checkpoint 对齐。

如果 `bridge` 判定观测 stale、推理超时或模式不是 `auto`，会发送 `hold` 命令包。
