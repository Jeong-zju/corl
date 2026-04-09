# 2026-04-09 `main/` Policy 部署到 ROS1 机器人的桥接方案

## 1. 目标

你的约束是：

- 训练好的 policy 来自 `main/` 下面这一套实现
- 真实机器人运行环境是 Ubuntu 20.04 + ROS1
- 不希望把 policy 直接写死在 ROS1 进程里
- 需要一个稳定的 `policy <-> bridge <-> ROS1` 通信框架

这份方案的核心目标是：

1. 让 policy 运行时与 ROS1 解耦
2. 让 ROS1 仍然掌握底层控制和安全权限
3. 让 bridge 成为唯一的数据协议转换层
4. 后续可以在不改 ROS1 侧代码的情况下替换 `act` / `streaming_act`

## 2. 从当前仓库确认到的事实

### 2.1 在线推理接口

`streaming_act` 当前在线推理入口是：

- `StreamingACTPolicy.select_action(batch)`
- `StreamingACTPolicy.reset()`

其中：

- `select_action()` 内部自己维护 action queue
- 如果启用了 visual prefix memory，也是在 policy 内部维护状态
- 因此 **policy runtime 必须是一个常驻进程**，不能每步重新起一个 Python 进程

### 2.2 当前 policy 是单步观测输入

配置里明确限制：

- `n_obs_steps = 1`
- `n_obs_steps != 1` 会直接报错

这意味着部署时：

- 不需要给 policy 维护多帧 observation window
- 但如果是 `streaming_act`，仍然需要在线维护历史相关量

### 2.3 当前真实数据的输入输出协议

从 `main/data/zeno-ai/wholebody_hanger_stage3_v30/meta/info.json` 可以确认，当前数据集特征是：

- `observation.state`: 17 维
- `action`: 17 维
- 图像：3 路相机，`224 x 224 x 3`，30 FPS

状态/动作维度顺序是：

```text
[base_vx, base_vy, base_omega,
 left_joint_0 ... left_joint_6,
 right_joint_0 ... right_joint_6]
```

视觉 key 是：

```text
observation.images.realsense_top
observation.images.realsense_left
observation.images.realsense_right
```

### 2.4 对 `streaming_act` 的额外要求

如果你部署的是 `streaming_act`，在线推理时除了当前图像和状态，还需要：

- `observation.path_signature`
- `observation.delta_signature`

但不需要把训练时的 `prefix_*` 整条序列喂进去。在线模式下：

- visual prefix memory 由 policy 自己维护
- bridge 只要在每个控制 tick 提供当前状态、当前图像、当前 signature 即可

### 2.5 一个非常重要的风险

当前数据转换脚本里：

- `action[:3]` 不是显式的 base command
- 它是直接从 odom/base velocity 拷贝过来的

也就是说，当前 `action` 前 3 维在语义上更像“观测到的底盘速度”，不是真正的“底盘控制标签”。

这会带来一个部署结论：

- 如果你现在直接把 policy 输出前 3 维当作底盘控制指令，风险很高
- 更稳妥的第一版部署应当优先做“机械臂主控，底盘固定/限权/人工接管”
- 如果你后续一定要让 policy 控底盘，建议补采真实 base action 并重训

### 2.6 需要按真实 embodiment 确认这 7 维的语义

当前导出的状态和动作在形式上是：

- 3 维底盘速度
- 左右臂各 7 维

但这 7 维在不同机器人上可能有两种语义：

- 7 个 arm joints
- 6 DoF arm + 1 gripper

对当前 zeno 平台，后续确认的真实语义是：

- 每臂 7 维 = `joint1 .. joint6 + gripper`

这意味着 bridge 和 ROS1 adapter 必须严格保持这个顺序，不要把最后 1 维误当成“第 7 个 arm joint”。

## 3. 推荐总体架构

推荐用 3 层结构：

```text
Policy Runtime  <---->  Bridge Core  <---->  ROS1 Adapter
```

职责划分如下：

### 3.1 Policy Runtime

职责：

- 加载 `main/` 训练出的 checkpoint
- 常驻持有 policy 对象
- 接收 bridge 送来的 observation packet
- 调用 `select_action()`
- 返回 action packet
- 接收 `reset/load/pause/resume/health` 等控制命令

要求：

- 不 import ROS
- 不关心 topic 名称
- 只关心“训练时的数据字段”和“模型前处理”

### 3.2 Bridge Core

职责：

- 汇总来自 ROS1 Adapter 的多模态观测
- 做时间对齐、丢帧判断、重采样、缩放、字段重排
- 组装成 policy 需要的 batch
- 如果是 `streaming_act`，在线计算 path signature 和 delta signature
- 把 action 拆成 ROS1 侧真正可执行的控制语义
- 做 watchdog、超时、限幅、模式切换

Bridge 是整个系统里唯一“既懂机器人协议，也懂 policy 输入输出协议”的部分。

### 3.3 ROS1 Adapter

职责：

- 订阅 ROS1 topic
- 从 ROS message 中抽取机器人状态和图像
- 把 bridge 返回的控制命令发布到 ROS1 控制接口
- 持有安全开关、E-stop、模式仲裁、controller enable/disable

要求：

- 不加载 PyTorch
- 不持有模型
- 即使 policy runtime 挂掉，ROS1 侧也仍然能安全降级

## 4. 推荐部署拓扑

### 4.1 推荐版本：双机部署

最推荐：

- 机器人控制机：Ubuntu 20.04 + ROS1 + ROS1 Adapter
- 推理机：独立 GPU 机，运行 Bridge Core + Policy Runtime

优点：

- 不污染 ROS1 环境
- CUDA / PyTorch / LeRobot 依赖可以独立升级
- GPU 推理不会影响 ROS1 主控机稳定性
- 后续换模型最方便

### 4.2 次优版本：单机双环境

如果必须单机部署，建议：

- 宿主机保留 ROS1
- policy runtime 放在独立 conda 环境或 Docker 容器
- bridge 通过 localhost/Unix socket 与 ROS1 Adapter 通信

关键原则仍然不变：

- ROS1 侧只做 ROS 和安全
- 模型推理不要塞进 `rospy` callback

## 5. 通信方式建议

### 5.1 推荐通信栈

建议：

- `Bridge Core <-> Policy Runtime`：ZeroMQ
- `ROS1 Adapter <-> Bridge Core`：同样用 ZeroMQ，或者同机时用 Unix Domain Socket

原因：

- 比把 policy 做成 ROS node 更解耦
- 比 rosbridge/websocket 更适合实时控制
- 跨机器和跨 Python 环境都方便
- 对 numpy / bytes 传输比较友好

如果你更偏向强 schema，也可以：

- header 用 protobuf
- 图像 payload 用原始 bytes 或 JPEG bytes

### 5.2 不建议的方式

不建议直接：

- 用 ROS topic 把图像和 tensor 一层层传到 policy
- 用 HTTP/Flask 做逐帧同步推理
- 用 rosbridge websocket 走实时控制闭环

这些方案都更容易在时延、依赖和稳定性上吃亏。

## 6. 推荐协议设计

### 6.1 Observation Packet

建议 bridge 发给 policy runtime 的观测包至少包含：

```python
{
    "seq": int,
    "episode_id": str,
    "stamp_ns": int,
    "reset": bool,
    "policy_type": "act" | "streaming_act",
    "state": float32[17],
    "images": {
        "realsense_top": uint8[224, 224, 3],
        "realsense_left": uint8[224, 224, 3],
        "realsense_right": uint8[224, 224, 3],
    },
    "path_signature": float32[signature_dim] | None,
    "delta_signature": float32[signature_dim] | None,
    "mode": "auto" | "hold" | "teleop",
}
```

其中：

- `act` 可以不带 signature
- `streaming_act` 必须带 `path_signature` 和 `delta_signature`
- `reset=true` 时，policy runtime 必须先调用 `policy.reset()`

### 6.2 Action Packet

建议 policy runtime 返回：

```python
{
    "seq": int,
    "obs_seq": int,
    "stamp_ns": int,
    "runtime_ms": float,
    "action": float32[17],
    "status": "ok" | "stale" | "error",
    "message": str,
}
```

### 6.3 Control Packet

建议单独保留控制通道，用于：

- `load_checkpoint`
- `reset_episode`
- `pause`
- `resume`
- `health_check`
- `shutdown`

这样观测流和控制流不会互相污染。

## 7. Bridge 里的关键逻辑

### 7.1 时间同步

ROS1 Adapter 需要把以下输入对齐到同一个控制 tick：

- top/left/right 三路相机
- 左臂 joint state
- 右臂 joint state
- base odom

建议做法：

- 以控制频率 30Hz 驱动主循环
- 每个 tick 读取最近时间窗内最新的一帧
- 如果任一关键数据超过阈值没更新，直接判定为 stale

推荐阈值：

- 图像最大陈旧时间：50 ms
- 关节状态最大陈旧时间：20 ms
- odom 最大陈旧时间：20 ms

### 7.2 字段重排

Bridge 要严格按训练时顺序构造 `state`：

```text
[base_vx, base_vy, base_omega,
 left_joint_0..6,
 right_joint_0..6]
```

顺序不能变。

### 7.3 图像前处理

建议在 bridge 内完成：

- resize 到训练分辨率
- RGB 排列统一
- uint8 -> float32 / tensor 的最终转换放在 policy runtime 内

原因：

- ROS1 Adapter 保持轻量
- policy runtime 内仍然能完全复用训练时前处理

### 7.4 `streaming_act` 的 signature 在线计算

如果部署 `streaming_act`，bridge 维护：

- `state_history`
- `previous_signature`

每步做：

1. 追加当前 `state`
2. 计算当前 `path_signature`
3. 计算 `delta_signature = current_signature - previous_signature`
4. 发送给 policy runtime

注意：

- 这一部分应该放在 bridge，不要塞进 ROS1 Adapter
- episode reset 或 safety reset 时必须清空历史

### 7.5 reset 语义

以下情况必须触发一次完整 reset：

- 新 episode 开始
- 任务切换
- 机器人 re-home
- E-stop 恢复后重新进入自主模式
- 相机/状态同步断裂超过阈值

reset 时应当同时清空：

- policy 内部 action queue
- visual prefix memory
- bridge 内的 state history
- previous signature

## 8. ROS1 侧动作解释

### 8.1 推荐的第一版动作映射

基于当前训练数据语义，推荐第一版这样拆 action：

- `action[3:10]` -> 左臂 7 维关节目标
- `action[10:17]` -> 右臂 7 维关节目标
- `action[0:3]` -> 默认先不直接用于底盘闭环控制

也就是说，第一版建议做：

- 双臂由 policy 控制
- 底盘固定、慢速限权，或由单独模块控制

### 8.2 如果你坚持让 policy 控底盘

可以做，但我不推荐直接上真实机器。因为当前数据里底盘 action 不是独立标注的真实控制量。

如果一定要试：

- 必须先在 bridge 内加极强限幅
- 必须有 watchdog 和人工接管
- 最好先在离线回放和半实物环境验证

更正统的做法是：

- 补采 base command topic
- 重新导出数据
- 重训 policy

### 8.3 gripper

如果你的 embodiment 和当前 zeno 一样，是每臂 `6 DoF + gripper`，那么：

- gripper 已经包含在每臂 7 维动作里
- bridge 不需要额外脑补 gripper 指令
- 但 ROS1 command topic 必须支持把 `joint1 .. joint6 + gripper` 一起下发

如果某个控制器不接受联合轨迹，而要求 gripper 单独 topic，那么就需要在 bridge 里再拆一层 command adapter。

## 9. 实时性与时延预算

目标控制频率建议先按 20 到 30 Hz 做。

一个比较稳妥的预算是：

- ROS 数据收集和对齐：5 到 10 ms
- bridge 前处理和 signature：3 到 8 ms
- policy inference：8 到 25 ms
- 命令发布：1 到 3 ms

总预算最好控制在：

- 单步 33 ms 以内用于 30 Hz
- 如果暂时做不到，就先降到 20 Hz

### 9.1 `n_action_steps` 的建议

对真实机器人部署，优先建议：

- `n_action_steps = 1`

原因：

- 当前 `main/bash/defaults/zeno-ai/.../streaming_act.yaml` 的 eval 也是这么设的
- 更安全
- 更容易处理延迟、reset 和人工接管

虽然 policy 内部支持 action queue，但真实系统先把每步闭环跑稳更重要。

## 10. 安全与降级机制

这一部分必须放在 ROS1/bridge 一侧，不能依赖 policy 自觉。

至少要有：

1. E-stop 永远高于 policy
2. joint limit clamp
3. base velocity clamp
4. stale observation 检测
5. stale action 检测
6. policy timeout watchdog
7. mode manager: `teleop / auto / hold`

推荐规则：

- 超过 2 个 control tick 没收到新 action：进入 `hold`
- 任一关键传感器 stale：停止向 policy 继续送新数据，并进入 `hold`
- policy runtime 挂掉：ROS1 继续活着，并维持安全姿态

## 11. 推荐的软件目录划分

如果后续在仓库里正式实现，建议新建：

```text
main/deploy/
  policy_runtime/
    server.py
    loader.py
    preprocess.py
  bridge/
    bridge_core.py
    sync.py
    signature_runtime.py
    protocol.py
  ros1_adapter/
    ros1_adapter_node.py
    ros_topics.py
    ros_publishers.py
  configs/
    deploy_zeno_streaming_act.yaml
```

这样职责会比较清晰：

- `policy_runtime/` 只面向模型
- `bridge/` 只面向协议和运行时
- `ros1_adapter/` 只面向 ROS1

## 12. 推荐落地顺序

### Phase 1

先只打通：

- ROS1 Adapter 读状态和图像
- Bridge 组 observation packet
- Policy Runtime 返回 17 维 action
- 先不真正下发到底盘和机械臂，只记录日志

### Phase 2

接着做：

- 双臂 joint target 下发
- 底盘仍然保持人工控制或固定
- 跑小范围安全测试

### Phase 3

稳定后再加：

- reset/pause/resume
- watchdog
- stale handling
- health monitor

### Phase 4

如果确认需要底盘自主，再做：

- base action 数据重构
- 重训
- 单独验证 base policy

## 13. 最终建议

对于你当前这套 `main/` policy，最稳妥、最贴近现状的方案是：

1. 保留 ROS1 作为底层控制和安全层
2. 把 policy 变成独立常驻的 inference service
3. 用 bridge 统一做时间同步、字段映射、signature 计算和安全降级
4. 第一版只让 policy 控双臂，不直接放开底盘前 3 维
5. 如果是 `streaming_act`，一定要在 bridge 中正确维护 `reset + path_signature + delta_signature`

一句话概括：

> 不要让 policy 知道 ROS，也不要让 ROS 知道模型细节；所有“训练协议”和“机器人协议”之间的差异，都由 bridge 吃掉。
