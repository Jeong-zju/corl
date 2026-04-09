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
- `action[3:10]`: 左臂 7 维关节目标
- `action[10:17]`: 右臂 7 维关节目标

如果 `bridge` 判定观测 stale、推理超时或模式不是 `auto`，会发送 `hold` 命令包。

