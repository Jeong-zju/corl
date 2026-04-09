# Deploy Bridge

`main/deploy/` 提供了一个三层部署骨架：

- `policy_runtime/`: 常驻加载 `main/` 训练出的 `act` / `streaming_act` checkpoint
- `bridge/`: 负责同步 ROS1 传感器流、在线计算 signature、请求 policy、拆解动作
- `ros1_adapter/`: 订阅 ROS1 topic 并发布标准 ROS1 控制消息

## 依赖

- `policy_runtime/`: `torch`, `lerobot`, `pyzmq`
- `bridge/`: `numpy`, `pyzmq`, 可选 `signatory`
- `ros1_adapter/`: `rospy`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `opencv-python`, `pyzmq`
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

如果要部署 vanilla `act`，把 `--policy-type` 改成 `act` 即可：

```bash
python3 main/deploy/policy_runtime/server.py \
  --policy-type act \
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

`act` 对应使用：

```bash
python3 main/deploy/bridge/bridge_core.py \
  --config main/deploy/configs/deploy_zeno_act.yaml
```

3. 在 ROS1 机器上启动 adapter
```bash
python3 main/deploy/ros1_adapter/ros1_adapter_node.py \
  --config main/deploy/configs/deploy_zeno_streaming_act.yaml
```

`act` 对应使用：

```bash
python3 main/deploy/ros1_adapter/ros1_adapter_node.py \
  --config main/deploy/configs/deploy_zeno_act.yaml
```

## 双路对比模式

如果你想同时启动 `act` 和 `streaming_act` 做 bag 回放对比，直接使用这两份并行配置：

- `main/deploy/configs/deploy_zeno_compare_act.yaml`
- `main/deploy/configs/deploy_zeno_compare_streaming_act.yaml`

它们已经做了这些隔离：

- 不同的 ZeroMQ 端口
- 不同的 ROS node name
- 不同的输出 topic 命名空间

默认输出 topic 是安全的对比命名空间，不会直接写到真实控制口：

- `act`: `/compare/act/left_arm_joint_states`, `/compare/act/right_arm_joint_states`, `/compare/act/cmd_vel`
- `streaming_act`: `/compare/streaming_act/left_arm_joint_states`, `/compare/streaming_act/right_arm_joint_states`, `/compare/streaming_act/cmd_vel`

示例启动命令：

```bash
python3 main/deploy/policy_runtime/server.py \
  --policy-type act \
  --policy-path <act_ckpt> \
  --device cuda \
  --bind tcp://*:5565 \
  --control-bind tcp://*:5568
```

```bash
python3 main/deploy/bridge/bridge_core.py \
  --config main/deploy/configs/deploy_zeno_compare_act.yaml
```

```bash
python3 main/deploy/ros1_adapter/ros1_adapter_node.py \
  --config main/deploy/configs/deploy_zeno_compare_act.yaml
```

```bash
python3 main/deploy/policy_runtime/server.py \
  --policy-type streaming_act \
  --policy-path <streaming_act_ckpt> \
  --device cuda \
  --bind tcp://*:5575 \
  --control-bind tcp://*:5578
```

```bash
python3 main/deploy/bridge/bridge_core.py \
  --config main/deploy/configs/deploy_zeno_compare_streaming_act.yaml
```

```bash
python3 main/deploy/ros1_adapter/ros1_adapter_node.py \
  --config main/deploy/configs/deploy_zeno_compare_streaming_act.yaml
```

也可以直接用一键启动脚本：

```bash
bash main/bash/start_deploy_compare.sh \
  --act-policy-path <act_ckpt> \
  --streaming-act-policy-path <streaming_act_ckpt>
```

常用可选参数：

- `--act-device cuda`
- `--streaming-act-device cuda`
- `--act-n-action-steps 1`
- `--streaming-act-n-action-steps 1`
- `--log-dir /tmp/corl_deploy_compare`

脚本会自动：

- 从对比版 YAML 读取端口
- 拉起 6 个进程
- 把日志分别写到 `--log-dir`
- 在任一进程退出时统一停掉整套对比链路

## 当前默认动作映射

- `action[0:3]`: 底盘
- `action[3:10]`: 左臂 7 维目标
- `action[10:17]`: 右臂 7 维目标

对于当前 zeno 配置，这两个 7 维目标的语义是：

- `joint1 .. joint6`
- `gripper`

也就是说，bridge 当前会把每只手臂的最后 1 维和前 6 个关节一起下发成 7 维目标。`ros1_adapter` 现在固定发布 `sensor_msgs/JointState`。

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

对于 `act`：

- bridge 不会在线计算 signature
- observation packet 不会携带 `path_signature` / `delta_signature`
- `deploy_zeno_act.yaml` 里的 `signature_*` 配置会被忽略

如果 `bridge` 判定观测 stale、推理超时或模式不是 `auto`，会发送 `hold` 命令包。
