# 2026-04-07 RMBench 双环境实现方案与 AI 编程 Prompt 整理

## 1. 讨论目标

本轮讨论的目标已经明确收敛为：

- `RMBench` 单环境方案已经确认不可行；
- 必须采用**双环境**方案：
  - `RMBench` 评测环境负责 simulator / evaluator client；
  - `corl-py312` 负责 `act` 与 `streaming_act` policy runtime；
- **不能修改任何 `RMBench` 代码**；
- 需要把方案整理成一套可落地的工程设计；
- 需要把实施过程拆成一组详细的 AI 编程 prompts，方便后续逐步让 AI 落代码。

这里的重点不是“怎样在 `RMBench` 里硬接一个 policy 文件”，而是：

- 如何在完全不改 `RMBench` 源码的前提下，
- 让 `RMBench` evaluator 与 `main` 侧 checkpoint / policy runtime 稳定协作，
- 并最终输出 `main` 风格的 `summary.json`、assert 结果和可追踪产物。

## 2. 已确认的关键事实

### 2.1 `RMBench` 推荐环境与主仓环境存在现实冲突

`RMBench` README 仍然要求单独 conda 环境，且推荐 `python=3.10`：

- `RMBench/README.md`

但当前目标 policy 所依赖的主工作环境是：

- `corl-py312`

因此，继续推进单环境已经没有收益。

### 2.2 `RMBench` 确实有 server/client 评测链路，但不能直接拿来当最终方案

仓库中已有：

- `RMBench/script/policy_model_server.py`
- `RMBench/script/eval_policy_client.py`
- `RMBench/policy/DP/eval_double_env.sh`
- `RMBench/policy/Your_Policy/eval_double_env.sh`
- `RMBench/policy/Your_Policy/deploy_policy_double_env.py`

这说明 `RMBench` evaluator 已具备“双环境”的基本思路：

- client 在 simulator 环境里跑；
- model server 在另一个环境里跑；
- 中间通过 socket 传输观测和动作。

但它不能直接成为最终方案，原因有两个：

1. `RMBench` 原有 policy 包默认仍从各自 `__init__.py -> deploy_policy.py` 导出；
2. server/client 的 config 解析器行为不一致，直接靠 `--overrides` 传复杂类型很脆弱。

因此，不能把目标定成“复用 `RMBench/policy/DP/eval_double_env.sh` 并小改几行”，因为这会落回修改 `RMBench` 的路线上。

### 2.3 `Your_Policy` 可以作为“接口模板参考”，但不能作为落地位置

`RMBench/policy/Your_Policy/` 这一组文件很重要，因为它实际上给出了 RMBench 官方希望用户接入自定义 policy 时的最小接口模板：

- `deploy_policy.yml`
  - 说明 server/client 共享一份 policy config
- `eval_double_env.sh`
  - 说明双环境启动顺序是：先 server，后 client
- `deploy_policy_double_env.py`
  - 说明 policy 侧至少应提供：
    - `encode_obs(observation)`
    - `get_model(usr_args)`
    - `eval(TASK_ENV, model, observation)`
  - 同时默认假设 server 上的 model 可能暴露：
    - `obs_cache`
    - `get_action`
    - `update_obs`

这对我们有两个直接帮助：

1. 它确认了双环境 policy bridge 的“官方接口形状”；
2. 它告诉我们 client 侧 `eval()` 应尽量保持轻量，只做 observation 编码、动作请求和环境执行。

但它不能直接作为最终落地位置，因为本轮有一个硬约束：

- **不能修改任何 `RMBench` 文件**

所以最终方案应当是：

- 参考 `Your_Policy` 的接口形状；
- 但把 bridge 模块实际放在 `main/benchmarks/RMBench/bridge/`；
- 再通过 `policy_name + PYTHONPATH + external config` 把这个外部 bridge 注入到 RMBench 的 server/client 链路中。

### 2.4 `main` 侧已有可复用的 checkpoint 加载与在线 eval 逻辑

当前主仓中已经有：

- `main/scripts/eval_policy.py`
- `main/scripts/eval_helpers.py`
- `main/policy/lerobot_policy_streaming_act/...`

这意味着：

- `act` checkpoint 的加载逻辑已经存在；
- `streaming_act` checkpoint 的加载逻辑已经存在；
- prefix sequence / path signature / delta signature / anchor / visual prefix memory 的在线构造逻辑已经存在；
- 真正缺的是“把 RMBench observation 适配到主仓 policy 输入”的 bridge，以及“双环境编排层”。

### 2.5 `main/data/rmbench` 已经给出了我们需要的输入语义

从 `main/data/rmbench/*/meta/info.json` 可确认，当前主仓处理后的 RMBench 数据集采用如下字段：

- `observation.state`: shape `(14,)`
- `observation.images.cam_high`
- `observation.images.cam_left_wrist`
- `observation.images.cam_right_wrist`
- `action`: shape `(14,)`

这非常重要，因为它给出了 bridge 的目标语义：

- `RMBench` 原始 observation 需要在线映射成这组 `main` 侧字段；
- 这样 bridge 才能尽量复用现有 `main` 的 preprocessor / checkpoint / policy forward。

## 3. 本轮讨论的核心结论

### 3.1 明确采用“RMBench client env + corl-py312 server env”

最终推荐架构为：

- `RMBench(py310)`：
  - 运行 `RMBench/script/eval_policy_client.py`
  - 负责 simulator、task rollout、success 统计、视频导出
- `corl-py312`：
  - 运行 `RMBench/script/policy_model_server.py`
  - 负责加载 `main` 侧 `act` / `streaming_act` checkpoint
  - 负责在线构造 policy 输入并输出动作

### 3.2 不修改 `RMBench`，而是从 `main/benchmarks/RMBench` 注入 bridge 和编排层

所有新增内容都应放到：

- `main/benchmarks/RMBench/`

而不是：

- `RMBench/policy/...`
- `RMBench/script/...`
- `RMBench/__init__...`

换句话说：

- `RMBench` 被视为**不可改 backend**；
- `main/benchmarks/RMBench` 是新的 **orchestration + bridge** 层。

### 3.3 不能依赖修改 `RMBench` 内置 policy package

因为不允许修改 `RMBench` 代码，所以不能采用：

- 改 `RMBench/policy/ACT/__init__.py`
- 新增 `RMBench/policy/ACT/deploy_policy_double_env.py` 并让 `__init__` 指向它
- 改 `RMBench` 自带 shell 脚本

替代方案是：

- 在 `main/benchmarks/RMBench/bridge/` 下创建**独立 bridge 模块**
- 用外部 config 把 `policy_name` 指向该 bridge 模块的 import path
- 运行 `RMBench` 脚本时通过 `PYTHONPATH` 暴露该模块

例如可使用：

- `policy_name: main.benchmarks.RMBench.bridge.act_bridge`
- `policy_name: main.benchmarks.RMBench.bridge.streaming_act_bridge`

这两者会被 `RMBench` 的 `importlib.import_module(policy_name)` 直接导入，而无需触碰 `RMBench/policy/*`。

### 3.4 同一个 bridge 模块必须同时适配 server 和 client 导入

这是整个方案最关键的工程约束之一。

因为 `RMBench` 的双环境脚本链路里：

- server 会按 `policy_name` 导入模块并调用 `get_model()`
- client 也会按同一个 `policy_name` 导入模块并调用 `eval()`

因此 bridge 模块必须满足：

1. 顶层 import 只能依赖**client env 也具备**的轻量依赖；
2. `torch / lerobot / streaming_act` 等重依赖必须**延迟到 `get_model()` 内部**再导入；
3. `eval()` 必须只做轻量 observation 编码与 socket 请求，不做重模型加载；
4. server 真正的策略状态应保存在 `get_model()` 返回的 runtime 对象里。

这里可以直接把 `RMBench/policy/Your_Policy/deploy_policy_double_env.py` 视为参考接口：

- 它的 `get_model()` 对应 server 初始化入口；
- 它的 `eval()` 对应 client 单步 rollout 入口；
- 它建议 observation 先经过 `encode_obs()` 再送入 model。

但我们不应完全照搬它的内部方法名设计。对于 `main` 侧 `act` / `streaming_act`，更稳的做法是：

- client 侧仍然保留 `encode_obs()` 语义；
- server 侧 runtime 统一暴露：
  - `select_action(raw_obs)`
  - `reset_model()`

而不是强行把 `main` policy 伪装成：

- `obs_cache`
- `update_obs`
- `get_action`

原因是：

- `main` 侧已有稳定的 `policy.select_action()` / `policy.reset()` 语义；
- 如果为了贴 `Your_Policy` 模板而额外制造一层假 `obs_cache/update_obs`，反而会增加状态同步复杂度；
- 对 `streaming_act full` 来说，prefix/signature/anchor 历史最好统一保存在 server runtime 内，而不是拆散成 client/server 两头维护。

### 3.5 streaming 特征应尽量在 server 端构造，而不是 client 端

对于 `streaming_act`，真正复杂的部分不是 socket，而是在线构造：

- path signature
- delta signature
- prefix sequence
- first-frame anchor
- visual prefix memory

这些逻辑应尽量都放在 `corl-py312` 的 server 端完成，原因是：

- 这样 client env 不需要安装额外依赖；
- 可以直接复用 `main/scripts/eval_helpers.py` 中的在线构造逻辑；
- 可以避免把 `signatory`、`lerobot`、`streaming_act` 依赖强行塞进 `RMBench` 环境；
- 可以让 client 只发送“原始观测载荷”，保持职责单纯。

### 3.6 必须用“生成后的 resolved config”，不能依赖大量 `--overrides`

当前 `RMBench` server/client 的 config parser 行为并不对称：

- client 侧 `--overrides` 会尝试 `eval(value)`
- server 侧 `--overrides` 仅对非常有限的数值场景做解析

因此如果我们直接依赖大量 `--overrides` 传：

- `bool`
- `list`
- `Path`
- `None`
- 复杂 feature 开关

很容易出现 server/client 解析不一致。

最终推荐做法是：

- 在 `main/benchmarks/RMBench` 编排层里为每次 run 生成一份**resolved YAML config**
- server 和 client 都只读取这份最终 config
- CLI 覆盖只保留极少数简单字符串或整数参数

这是稳定性要求，不是“实现风格偏好”。

## 4. 推荐总体架构

### 4.1 目录建议

推荐新增如下结构：

```text
main/benchmarks/RMBench/
  __init__.py
  README.md
  doctor.py
  run_dual_env.py
  assert_summary.py
  config_templates/
    act_dual_env.base.yml
    streaming_act_dual_env.base.yml
  bridge/
    __init__.py
    common.py
    runtime.py
    act_bridge.py
    streaming_act_bridge.py
```

说明：

- `doctor.py`：做环境检查与路径检查
- `run_dual_env.py`：唯一官方双环境入口
- `assert_summary.py`：对 `summary.json` 做阈值断言
- `config_templates/`：只存模板，不存每次 run 的最终配置
- `bridge/`：server/client 共同导入的 policy bridge 模块

### 4.2 运行时数据流

推荐数据流如下：

1. 用户从 `main/benchmarks/RMBench/run_dual_env.py` 发起评测；
2. 编排层解析参数并创建 canonical run 目录；
3. 编排层生成 resolved config YAML；
4. 编排层选择空闲端口；
5. 编排层在 `corl-py312` 中启动：
   - `RMBench/script/policy_model_server.py`
6. 编排层在 `RMBench(py310)` 中启动：
   - `RMBench/script/eval_policy_client.py`
7. client 从 simulator 获取 observation；
8. client bridge 将 observation 编码为轻量 raw payload，经 socket 发给 server；
9. server bridge 把 raw payload 转成主仓 policy 输入，运行 `policy.select_action()`；
10. server 返回动作；
11. client 把动作喂给 `TASK_ENV.take_action()`；
12. rollout 完成后，编排层定位 `RMBench/eval_result/...` 中的原始结果目录；
13. 编排层将结果归档到 `main/outputs/eval/...` 的 canonical run 目录，并写 `summary.json`。

### 4.3 结果目录建议

为了兼容主仓 monitor 与现有 `outputs/eval/{env}/{algorithm}/{timestamp}` 风格，建议 canonical 输出目录使用：

```text
main/outputs/eval/rmbench_<task_name>/<policy_series>/<run_tag>/
```

例如：

```text
main/outputs/eval/rmbench_cover_blocks/act/20260408_143011/
main/outputs/eval/rmbench_cover_blocks/streaming-act-sipm/20260408_151422/
```

目录内建议至少包含：

- `summary.json`
- `resolved_config.yml`
- `server.log`
- `client.log`
- `launcher.log`
- `raw_result.txt`
- `raw_eval_log.txt`，若存在
- `raw_eval_dir.txt`
- 可选：`videos/` 指向 RMBench 原始视频目录的符号链接

### 4.4 RMBench 原始结果与 canonical 结果的关系

因为不能修改 `RMBench` 代码，所以不能直接让它输出到 `main/outputs/eval/...`。

因此需要明确：

- `RMBench/eval_result/...` 是**原始产物目录**
- `main/outputs/eval/...` 是**主仓 canonical 目录**

编排层负责：

- 找到本次 run 对应的 `RMBench/eval_result/...`
- 复制或链接必要文件
- 解析 `_result.txt` 与日志
- 写出标准化 `summary.json`

## 5. 关键接口约定

### 5.1 bridge 模块对 RMBench 暴露的接口

每个 bridge 模块都至少需要：

- `get_model(usr_args)`
- `eval(TASK_ENV, model, observation)`

其中：

- `get_model()` 在 server 进程中执行，返回 runtime 对象；
- `eval()` 在 client 进程中执行，把当前 observation 转成 raw payload 并调用 `model.call("select_action", payload)`。

server runtime 对象至少需要实现：

- `select_action(raw_obs)`
- `reset_model()`

这一设计是“参考 `Your_Policy/deploy_policy_double_env.py` 的模块级接口，但不强行沿用其 runtime 方法名”的折中结果。

### 5.2 raw observation payload 的标准形态

推荐在 socket 上传输如下原始载荷：

```python
{
    "images": {
        "cam_high": np.ndarray,         # HWC, uint8
        "cam_left_wrist": np.ndarray,   # HWC, uint8
        "cam_right_wrist": np.ndarray,  # HWC, uint8
    },
    "state": np.ndarray,                # (14,), float32
    "instruction": str | None,          # 先允许保留，但当前 act/streaming_act 可忽略
}
```

映射规则应固定为：

- `head_camera.rgb -> observation.images.cam_high`
- `left_camera.rgb -> observation.images.cam_left_wrist`
- `right_camera.rgb -> observation.images.cam_right_wrist`
- `joint_action[left_arm + left_gripper + right_arm + right_gripper] -> observation.state`

### 5.3 server 侧 runtime 的职责边界

server runtime 必须负责：

- checkpoint 解析与加载
- preprocessor / postprocessor 初始化
- observation tensor 构造
- `streaming_act` 在线历史维护
- `select_action()` 推理
- `reset_model()` 清空 runtime 状态

client 不应负责：

- checkpoint 加载
- path signature 计算
- prefix memory 维护
- anchor 缓存
- 任何 `lerobot`/`torch` 相关逻辑

## 6. 推荐分阶段实施顺序

推荐严格按以下顺序推进：

1. 编排骨架与 doctor
2. 共享 runtime 与 config 生成
3. `act` bridge
4. `streaming_act baseline` bridge
5. `streaming_act full` bridge
6. 结果标准化与 assert
7. 文档和 smoke tests

不要跳过 `act` 和 `streaming_act baseline` 直接做 full，因为 full 模式最容易把问题混在一起：

- checkpoint 加载问题
- 观测映射问题
- socket 通信问题
- prefix/anchor/signature 在线构造问题

## 7. AI 实施的共享约定

以下 prompts 共用这些硬约束：

1. 不允许修改 `/home/jeong/zeno/corl/RMBench` 下任何文件。
2. 所有新增代码都放在 `/home/jeong/zeno/corl/main/benchmarks/RMBench` 下。
3. 不允许复制一份 `RMBench` 再修改；必须复用原仓脚本原样运行。
4. 不允许要求用户手动改 `PYTHONPATH` 或手动切环境；编排脚本必须自己处理。
5. bridge 模块顶层不得导入 `torch`、`lerobot`、`lerobot_policy_streaming_act` 等 client env 可能缺失的依赖。
6. 所有重依赖都必须延迟到 server 侧 `get_model()` 或 runtime 初始化阶段。
7. 必须使用生成后的 resolved config YAML，避免 server/client parser 类型不一致。
8. 所有 run 都必须输出 canonical `summary.json` 到 `main/outputs/eval/...`。
9. 默认使用 `conda run -n <env>` 启动 server/client，而不是依赖交互 shell 激活态。
10. 所有路径处理都必须考虑 `RMBench` 原始结果目录中含空格和冒号的时间戳目录名，禁止用容易被 shell 错切的字符串拼接。
11. 所有新代码都要提供至少一个最小 smoke test 命令。
12. 任何尚未支持的特性都必须显式报错，不能静默退化。

## 8. Coding Prompts

### 0. 共享 Prompt：双环境总约束

```text
你在 `/home/jeong/zeno/corl` 仓库中工作。当前目标是把 `main` 侧的 `act` 与 `streaming_act` checkpoint 接到 `RMBench` 做 benchmark eval，但有一个强硬约束：**不能修改任何 `RMBench` 代码**。`RMBench` 只能被当成不可改 backend 使用。

本任务的固定双环境架构是：
- simulator/client 环境：`RMBench` 对应 conda 环境，例如 `RMBench` 或 `RMBench-py310`
- policy/server 环境：`corl-py312`

必须遵守以下约束：
1. 不允许修改 `/home/jeong/zeno/corl/RMBench` 下任何文件。
2. 所有新增代码统一放到 `/home/jeong/zeno/corl/main/benchmarks/RMBench/`。
3. 必须复用以下 RMBench 原脚本，不得改动：
   - `RMBench/script/policy_model_server.py`
   - `RMBench/script/eval_policy_client.py`
4. 不允许通过修改 `RMBench/policy/*/__init__.py` 的方式接入 bridge。
5. 必须通过外部 bridge 模块 + 外部 config + `PYTHONPATH` 注入的方式完成接入。
6. 所有复杂配置都必须通过“生成后的 resolved YAML config”传给 server/client；不要依赖大量 `--overrides`。
7. bridge 模块必须同时能被 server 和 client 导入，因此：
   - 顶层 import 只能依赖 stdlib / numpy / 轻量库；
   - `torch` / `lerobot` / `streaming_act` 必须延迟到 `get_model()` 内部；
   - client 侧 `eval()` 只能做 observation 编码和 socket 调用；
   - server 侧 runtime 负责 checkpoint 加载与在线推理。
8. canonical 输出目录必须落到 `main/outputs/eval/...`，并写出 `summary.json`。
9. 所有新实现都必须有清晰的失败信息和 smoke test。

开始前先阅读这些文件：
- `RMBench/script/policy_model_server.py`
- `RMBench/script/eval_policy_client.py`
- `RMBench/policy/Your_Policy/eval_double_env.sh`
- `RMBench/policy/Your_Policy/deploy_policy_double_env.py`
- `main/scripts/eval_policy.py`
- `main/scripts/eval_helpers.py`
- `main/data/rmbench/classify_blocks/meta/info.json`
- `main/discussions/2026-04-07_03_rmbench_benchmark_orchestration_discussion.md`

最终回复必须包含：
- 修改文件列表
- 本阶段实现了什么
- 新增 CLI / config / 输出目录约定
- 最小 smoke test
- 尚未覆盖的风险点
```

### 1. Prompt 1：搭建 orchestration 骨架与 doctor

```text
你在 `/home/jeong/zeno/corl` 仓库中工作。请在不修改任何 `RMBench` 文件的前提下，为 RMBench 双环境评测搭建 orchestration 骨架，并实现一个环境与路径 doctor。

你必须先阅读：
- `main/discussions/2026-04-07_03_rmbench_benchmark_orchestration_discussion.md`
- `RMBench/script/policy_model_server.py`
- `RMBench/script/eval_policy_client.py`
- `RMBench/policy/Your_Policy/eval_double_env.sh`
- `RMBench/policy/Your_Policy/deploy_policy_double_env.py`
- `main/README.md`

任务目标：
1. 创建新的目录结构：
   - `main/benchmarks/RMBench/__init__.py`
   - `main/benchmarks/RMBench/README.md`
   - `main/benchmarks/RMBench/doctor.py`
   - `main/benchmarks/RMBench/config_templates/`
   - `main/benchmarks/RMBench/bridge/`
2. `doctor.py` 至少检查：
   - `RMBench` 根目录存在
   - `RMBench/script/policy_model_server.py` 存在
   - `RMBench/script/eval_policy_client.py` 存在
   - `main/scripts/eval_policy.py` 存在
   - `corl-py312` 名称已配置为 policy 环境
   - RMBench client 环境名称可通过 CLI 指定
   - `main/data/rmbench/collection.json` 存在
3. `doctor.py` 必须提供清晰的 CLI，例如：
   - `python main/benchmarks/RMBench/doctor.py --rmbench-root ... --policy-env corl-py312 --client-env RMBench`
4. doctor 输出必须包括：
   - 通过项
   - 失败项
   - 建议修复动作
5. `README.md` 先写清楚本方案的核心约束：
   - 不改 RMBench
   - 双环境
   - bridge 模块位置
   - canonical 输出目录位置
6. 不要在这一步实现真正的 eval 启动逻辑，但要为后续入口留出清晰命名。

实现要求：
1. 所有路径默认基于仓库根目录自动推断，但允许 CLI 覆盖。
2. doctor 退出码必须有语义：
   - 全部通过返回 0
   - 任一关键检查失败返回非 0
3. 代码里不要硬编码用户 home 目录，使用 `Path(__file__).resolve()` 推导仓库根。
4. 如果检查 conda 环境名称是否存在不够稳定，可以退一步只检查名称参数非空，并在输出里明确“未做激活验证”；但优先尝试做真实检查。

验收标准：
1. `main/benchmarks/RMBench/` 基本骨架已经建立。
2. doctor 至少能给出路径与环境参数层面的有效反馈。
3. README 已经说明为什么不改 RMBench，以及为什么采用 bridge + external config。

最终回复必须包含：
- 新增文件列表
- doctor 的 CLI 用法
- 一条成功 smoke test 命令
- 一条故意失败的 smoke test 命令
```

### 2. Prompt 2：实现 resolved config 生成与运行参数模型

```text
你在 `/home/jeong/zeno/corl` 仓库中工作。当前已经有 `main/benchmarks/RMBench` 骨架。请继续实现“每次 run 生成一份 resolved YAML config”的机制，作为后续 server/client 的唯一配置输入。

开始前先阅读：
- `RMBench/script/policy_model_server.py`
- `RMBench/script/eval_policy_client.py`
- `RMBench/policy/ACT/deploy_policy.yml`
- `main/discussions/2026-04-07_03_rmbench_benchmark_orchestration_discussion.md`

任务目标：
1. 新增基础模板：
   - `main/benchmarks/RMBench/config_templates/act_dual_env.base.yml`
   - `main/benchmarks/RMBench/config_templates/streaming_act_dual_env.base.yml`
2. 新增 Python 代码，负责：
   - 解析用户 CLI 参数
   - 选择模板
   - 填充本次 run 的 resolved config
   - 把 resolved config 写到 canonical run 目录，例如：
     `main/outputs/eval/rmbench_<task>/<policy_series>/<run_tag>/resolved_config.yml`
3. resolved config 必须至少包含：
   - `policy_name`
   - `task_name`
   - `task_config`
   - `ckpt_setting`
   - `seed`
   - `instruction_type`
   - `policy_type`
   - `policy_path`
   - `device`
   - `n_action_steps`
   - `signature_backend`
   - 所有 streaming 相关显式开关
4. 这一步必须明确 bridge import path，例如：
   - `main.benchmarks.RMBench.bridge.act_bridge`
   - `main.benchmarks.RMBench.bridge.streaming_act_bridge`
5. 不能依赖 `--overrides` 传布尔、列表、Path 等复杂类型；resolved config 必须是 server/client 的主输入。

实现要求：
1. 配置模型最好使用 dataclass 或等价结构，避免散乱 dict。
2. 如果某些字段只被 server 使用，也允许写入同一份 config 中；client 可以忽略。
3. `policy_path` 必须允许指向：
   - 训练 run 目录
   - `pretrained_model/`
   - `checkpoints/last/pretrained_model/`
4. 这一步不需要真正启动 server/client，但必须把最终 config 的 schema 固定下来。
5. 写出的 YAML 必须保留正确的数据类型，后续 server/client 读取后不应发生布尔与字符串混淆。

建议补一个小的 helper：
- 生成 `run_tag`
- 规范化 `policy_series`
- 规范化 canonical 输出目录

验收标准：
1. 能生成 `act` 的 resolved config。
2. 能生成 `streaming_act` 的 resolved config。
3. 同一个任务、不同 policy 的输出目录风格统一。
4. 不需要改 RMBench parser 也能承载完整配置。

最终回复必须包含：
- 修改文件列表
- resolved config 的主要字段
- 一个生成 act config 的命令
- 一个生成 streaming_act config 的命令
- 仍待后续阶段实现的部分
```

### 3. Prompt 3：实现共享的 LeRobot policy runtime

```text
你在 `/home/jeong/zeno/corl` 仓库中工作。请在不修改任何 `RMBench` 文件的前提下，实现一个共享的 server-side policy runtime，用于加载 `main` 侧 `act` / `streaming_act` checkpoint，并暴露统一的 `select_action(raw_obs)` 与 `reset_model()` 接口。

开始前先阅读：
- `main/scripts/eval_policy.py`
- `main/scripts/eval_helpers.py`
- `main/policy/lerobot_policy_streaming_act/src/lerobot_policy_streaming_act/...`
- `main/data/rmbench/classify_blocks/meta/info.json`
- `main/discussions/2026-04-07_03_rmbench_benchmark_orchestration_discussion.md`

任务目标：
1. 在 `main/benchmarks/RMBench/bridge/` 下新增共享 runtime 实现，例如：
   - `common.py`
   - `runtime.py`
2. runtime 需要统一完成：
   - checkpoint 路径解析
   - `act` / `streaming_act` config 加载
   - preprocessor / postprocessor 初始化
   - raw RMBench observation -> main policy input 的在线转换
   - `policy.select_action()` 调用
   - `policy.reset()` 与 runtime history reset
3. 推荐统一对外暴露一个类，例如：
   - `DualEnvLeRobotRuntime`
4. 这个 runtime 至少支持 `policy_type in {"act", "streaming_act"}`。

实现要求：
1. 顶层模块 import 不允许依赖 client 环境不一定具备的重库。
2. 所有重依赖必须延迟到 runtime 初始化内部导入：
   - `torch`
   - `lerobot`
   - `lerobot_policy_streaming_act`
3. 必须复用主仓已有的 checkpoint 加载逻辑，而不是手写另一套权重恢复。
4. 必须复用主仓 pre/post processor 逻辑，而不是手写归一化。
5. raw observation 的最小输入契约固定为：
   - `images.cam_high`
   - `images.cam_left_wrist`
   - `images.cam_right_wrist`
   - `state`
6. 必须校验 checkpoint config 与输入特征：
   - 若 visual keys 不是预期的三路相机，则显式报错；
   - 若 state 维度不是 14，则显式报错；
   - 若 action 维度不是 14，则显式报错。
7. 先只实现“当前帧输入 -> select_action”，不要在这一步加入 path signature / prefix sequence / anchor / visual prefix memory。
8. `reset_model()` 必须清空 policy 自身缓存；若 `policy` 提供 `reset()`，就调用它。

建议实现的 helper：
1. `ensure_streaming_act_importable_for_runtime()`
2. `resolve_policy_dir_for_runtime()`
3. `build_rmbench_obs_tensors()`
4. `tensor_to_numpy_action()`

验收标准：
1. 可以通过 `policy_type=act` 加载一个主仓 checkpoint。
2. 可以通过 `policy_type=streaming_act` 加载一个主仓 checkpoint。
3. 在 server 侧可对一个 mock raw observation 调用 `select_action()` 并得到 shape `(14,)` 的动作。
4. 关闭 streaming 特性时，runtime 不会尝试访问 prefix/signature 相关字段。

最终回复必须包含：
- 修改文件列表
- runtime 的公共接口
- raw observation 到 policy 输入的字段映射
- 一个最小的本地 Python smoke test
- 已知限制
```

### 4. Prompt 4：实现 `act` 的无侵入 bridge

```text
你在 `/home/jeong/zeno/corl` 仓库中工作。请基于已实现的共享 runtime，为 RMBench 双环境评测实现 `act` bridge。强约束：不能修改任何 `RMBench` 文件。

开始前先阅读：
- `RMBench/script/eval_policy_client.py`
- `RMBench/script/policy_model_server.py`
- `RMBench/policy/Your_Policy/deploy_policy_double_env.py`
- `main/benchmarks/RMBench/bridge/runtime.py`
- `main/data/rmbench/classify_blocks/meta/info.json`
- `main/discussions/2026-04-07_03_rmbench_benchmark_orchestration_discussion.md`

任务目标：
1. 新增 bridge 模块：
   - `main/benchmarks/RMBench/bridge/act_bridge.py`
2. 模块必须能同时被 server 和 client 导入。
3. 模块至少导出：
   - `get_model(usr_args)`
   - `eval(TASK_ENV, model, observation)`
4. `get_model()`：
   - 在 server 环境中创建共享 runtime
   - 固定 `policy_type="act"`
5. `eval()`：
   - 在 client 环境中运行
   - 把 RMBench 原始 observation 编码为 raw payload
   - 通过 `model.call("select_action", payload)` 请求 server 动作
   - 调用 `TASK_ENV.take_action(action)`

实现要求：
1. `eval()` 顶层不允许导入 `torch`/`lerobot`。
2. `eval()` 不允许计算 signature、prefix、anchor，也不允许维护复杂历史。
3. action 必须是一维 `(14,)`，否则显式报错。
4. observation 编码时必须固定映射：
   - `head_camera -> cam_high`
   - `left_camera -> cam_left_wrist`
   - `right_camera -> cam_right_wrist`
   - joint state -> `observation.state`
5. 若 observation 中缺相机或关节字段，必须显式报错。
6. 为避免行为不一致，本 bridge 采用“每个 env step 向 server 请求一次 `select_action`”的模式，不在 client 侧做 action chunk 执行缓存。
7. 允许 `instruction` 暂时透传但不使用；不要强行引入文本编码。
8. 可以参考 `Your_Policy/deploy_policy_double_env.py` 的模块接口布局，但不要为了模仿模板而制造 `obs_cache/get_action/update_obs` 伪接口。

验收标准：
1. server/client 都能成功 import 此 bridge。
2. server 侧能创建 `act` runtime。
3. client 侧 `eval()` 能把一个 mock observation 编码并发起 socket 请求。
4. 对非法 observation/action shape 有清晰报错。

最终回复必须包含：
- 修改文件列表
- `act_bridge` 的导出接口
- raw payload schema
- 一个 import smoke test
- 一个 mock observation smoke test
```

### 5. Prompt 5：实现 `streaming_act baseline` bridge

```text
你在 `/home/jeong/zeno/corl` 仓库中工作。请基于共享 runtime，为 RMBench 双环境评测实现 `streaming_act baseline` bridge。这里的 baseline 特指：

- 不启用 path signature
- 不启用 delta signature
- 不启用 prefix sequence training
- 不启用 visual prefix memory
- 不启用 first-frame anchor

也就是说，这一步只验证：

`RMBench raw observation -> corl-py312 streaming_act checkpoint -> select_action -> RMBench rollout`

开始前先阅读：
- `main/benchmarks/RMBench/bridge/runtime.py`
- `main/scripts/eval_policy.py`
- `main/policy/lerobot_policy_streaming_act/...`
- `RMBench/policy/Your_Policy/deploy_policy_double_env.py`
- `main/discussions/2026-04-07_03_rmbench_benchmark_orchestration_discussion.md`

任务目标：
1. 新增：
   - `main/benchmarks/RMBench/bridge/streaming_act_bridge.py`
2. 支持与 `act_bridge` 相同的双导入模式：
   - server 侧 `get_model()`
   - client 侧 `eval()`
3. 在这一步里，server runtime 必须显式检查 checkpoint config：
   - 如果启用了 path signature / delta signature / prefix / anchor / visual prefix memory 任一特性，则直接报错；
   - 不允许静默忽略。
4. `eval()` 仍然保持轻量，只发原始 raw payload 给 server。

实现要求：
1. `streaming_act_bridge` 要尽量与 `act_bridge` 共享 observation 编码逻辑，避免重复。
2. server runtime 中必须明确区分：
   - `policy_type="streaming_act"`
   - baseline 模式
3. 如果 checkpoint config 中 `n_action_steps > 1`，允许保留，但需要在日志中提示；
   - 不要在这一步实现复杂的 online memory 同步讨论。
4. `reset_model()` 仍需调用 policy 自身 reset。

验收标准：
1. `streaming_act baseline` checkpoint 可以被 server 成功加载。
2. 如果 checkpoint 打开了 full 模式特性，本阶段会明确报错而不是偷偷忽略。
3. client/server 与 `act_bridge` 一样可以完成 raw observation 的单步请求。

最终回复必须包含：
- 修改文件列表
- baseline 支持的特性范围
- 对不支持特性的报错策略
- 一个 baseline smoke test
- 后续 full 模式还缺什么
```

### 6. Prompt 6：实现 `streaming_act full` 在线特征构造

```text
你在 `/home/jeong/zeno/corl` 仓库中工作。当前 `streaming_act baseline` 已经可用。请继续把 RMBench 双环境 bridge 扩展为支持 `streaming_act full`，并且依然不能修改任何 `RMBench` 文件。

开始前先阅读：
- `main/scripts/eval_policy.py`
- `main/scripts/eval_helpers.py`
- `main/policy/lerobot_policy_streaming_act/...`
- `main/benchmarks/RMBench/bridge/runtime.py`
- `main/benchmarks/RMBench/bridge/streaming_act_bridge.py`
- `main/discussions/2026-04-07_03_rmbench_benchmark_orchestration_discussion.md`

任务目标：
1. 在 server runtime 中，为 `streaming_act` 支持以下在线输入构造：
   - path signature
   - delta signature
   - prefix sequence
   - first-frame anchor
   - visual prefix memory 所需的在线历史缓存
2. 所有这些逻辑都必须放在 server 侧完成，client 仍只发送 raw observation。
3. 必须尽量复用 `main/scripts/eval_policy.py` 与 `main/scripts/eval_helpers.py` 的现有在线构造逻辑，而不是另写一套新逻辑。

实现要求：
1. runtime 必须维护 episode 级状态，例如：
   - `state_history`
   - `prefix_state_history`
   - `prefix_image_histories`
   - `prefix_signature_history`
   - `prefix_delta_signature_history`
   - `previous_signature_vec`
   - `first_frame_anchor`
2. `reset_model()` 必须清空上述所有历史，并重置 policy 内部状态。
3. path signature 的 backend 选择必须兼容主仓现有逻辑：
   - `auto`
   - `signatory`
   - `simple`
4. 若 checkpoint config 启用了某个 streaming 特性，但运行时缺少对应字段或 shape 不匹配，必须显式报错。
5. 若启用 visual prefix memory，建议默认把 `n_action_steps` 收紧到 `1`，或至少发出明确警告。
6. 必须保持与主仓 online eval 的语义一致：
   - first-frame anchor 是 reset 后第一帧
   - prefix 序列按时间从旧到新
   - 不允许未来泄漏
7. 不要把这些 streaming 计算放到 client 侧。

建议做法：
1. 把“当前 raw observation -> obs dict -> preprocessor -> streaming 补充字段 -> select_action”的流程封装为一个清晰方法。
2. 尽量调用现有 helper：
   - `build_prefix_sequence_eval_inputs`
   - `ensure_prefix_sequence_batch_dims`
   - `compute_signatory_signature_np`
   - `compute_simple_signature_np`
   - `compute_delta_signature_step_np`
3. 对 `streaming_act full` 与 baseline 的行为边界写清楚，避免混用。

验收标准：
1. full 模式 checkpoint 在 server 侧可以完成一次单步 select_action。
2. `reset_model()` 后历史缓存会被清空。
3. anchor / prefix / signature 的 shape 和 mask 有断言保护。
4. baseline 模式仍不受影响。

最终回复必须包含：
- 修改文件列表
- full 模式新增的 runtime state
- online 构造链路概览
- 一个最小 full 模式 smoke test
- 已知风险
```

### 7. Prompt 7：实现双环境 launcher 与结果标准化

```text
你在 `/home/jeong/zeno/corl` 仓库中工作。当前 bridge 已基本具备。请实现最终的双环境 launcher，使用户只需要调用一个 `main/benchmarks/RMBench` 入口，就能：

1. 生成 resolved config
2. 启动 policy server
3. 启动 RMBench client
4. 清理进程
5. 找到 RMBench 原始结果目录
6. 归档产物到 `main/outputs/eval/...`
7. 生成标准化 `summary.json`

强约束：不能修改任何 `RMBench` 文件。

开始前先阅读：
- `RMBench/script/policy_model_server.py`
- `RMBench/script/eval_policy_client.py`
- `RMBench/policy/Your_Policy/eval_double_env.sh`
- `main/scripts/eval_helpers.py`
- `main/benchmarks/RMBench/config_templates/...`
- `main/benchmarks/RMBench/bridge/...`
- `main/discussions/2026-04-07_03_rmbench_benchmark_orchestration_discussion.md`

任务目标：
1. 实现统一入口，例如：
   - `main/benchmarks/RMBench/run_dual_env.py`
2. CLI 至少支持：
   - `--rmbench-root`
   - `--client-env`
   - `--policy-env`
   - `--policy act|streaming_act`
   - `--task-name`
   - `--task-config`
   - `--ckpt-setting`
   - `--policy-path`
   - `--device`
   - `--seed`
   - `--signature-backend`
   - `--n-action-steps`
3. launcher 必须：
   - 选择空闲端口
   - 生成 resolved config
   - 在 `corl-py312` 中以 `cwd=RMBench root` 启动 `policy_model_server.py`
   - 在 client env 中以 `cwd=RMBench root` 启动 `eval_policy_client.py`
   - 给两个进程注入合适的 `PYTHONPATH`
4. 进程退出后，launcher 必须：
   - 可靠地终止 server
   - 解析 client stdout / log，定位本次 `RMBench/eval_result/...` 原始结果目录
   - 把关键产物复制或链接到 canonical run 目录
   - 写出标准化 `summary.json`

实现要求：
1. 推荐使用 `subprocess.Popen([...], cwd=..., env=...)`，不要拼接复杂 shell 字符串。
2. 推荐使用 `conda run -n <env>` 启动进程。
3. 所有 stdout/stderr 都要落文件，例如：
   - `server.log`
   - `client.log`
   - `launcher.log`
4. 需要考虑 `RMBench` 原始目录名里包含空格与冒号，路径解析必须稳健。
5. 若 client 输出里出现：
   - `Data has been saved to .../_result.txt`
   优先直接从该行解析原始结果路径，而不是靠“猜最新目录”。
6. 若无法直接解析路径，再退回到按 task/policy/task_config/ckpt_setting 搜索最新目录。
7. `summary.json` 至少包含：
   - `benchmark`
   - `env_type`
   - `task_name`
   - `task_config`
   - `policy_type`
   - `policy_series`
   - `success_rate`
   - `successes`
   - `episodes`
   - `seed`
   - `policy_path`
   - `device`
   - `client_env`
   - `policy_env`
   - `raw_eval_dir`
   - `resolved_config`
8. 如果 `_result.txt` 只有 success rate 而没有 successes/episodes，就从 client log 的最终 success line 解析；再不行时才退化为：
   - `episodes = 100`
   - `successes = round(success_rate * 100)`
   并在 summary 中标明是 estimated。

验收标准：
1. 一条命令能完成双环境启动与清理。
2. canonical 输出目录结构稳定。
3. `summary.json` 能被主仓 monitor / compare 工具消费。
4. 原始 RMBench 结果目录能被追踪回溯。

最终回复必须包含：
- 修改文件列表
- launcher 的 CLI
- canonical 目录结构
- 一条完整的 act 评测命令
- 一条完整的 streaming_act 评测命令
```

### 8. Prompt 8：实现 summary assert 与最终 README

```text
你在 `/home/jeong/zeno/corl` 仓库中工作。当前双环境 launcher 已能产出 canonical `summary.json`。请补齐最后一层工程收口：

1. success rate assert
2. RMBench benchmark README
3. 最小 smoke tests 文档

仍然不能修改任何 `RMBench` 文件。

开始前先阅读：
- `main/benchmarks/RMBench/run_dual_env.py`
- `main/benchmarks/RMBench/README.md`
- `main/scripts/compare_eval_summaries.py`
- `main/monitor/README.md`
- `main/discussions/2026-04-07_03_rmbench_benchmark_orchestration_discussion.md`

任务目标：
1. 实现：
   - `main/benchmarks/RMBench/assert_summary.py`
2. 该脚本至少支持：
   - `--summary <path>`
   - `--min-success-rate <float>`
   - `--require-raw-artifacts`
3. 若断言失败，必须返回非 0。
4. README 需要说明：
   - 双环境总体架构
   - 为什么不改 RMBench
   - `act` 命令
   - `streaming_act baseline` 命令
   - `streaming_act full` 命令
   - `summary.json` 与原始 `RMBench/eval_result` 的关系
   - 已知限制
5. README 需要明确给出最小 smoke test 顺序：
   - doctor
   - act import / runtime smoke test
   - streaming baseline smoke test
   - end-to-end dual-env smoke test

实现要求：
1. `assert_summary.py` 的字段读取要兼容你前一步写出的 summary schema。
2. 若 `summary.json` 中存在 `estimated_successes=true` 或等价字段，assert 输出中要提醒用户该结果来自估算而非明确 episode 计数。
3. README 中命令必须全部使用 `main/benchmarks/RMBench/run_dual_env.py` 统一入口，不要让用户直接手写 server/client 两条命令。
4. README 要明确标出：
   - 哪些产物在 `main/outputs/eval/...`
   - 哪些原始产物仍在 `RMBench/eval_result/...`

验收标准：
1. summary assert 可在成功和失败两种场景下给出正确退出码。
2. README 足以让仓库使用者从零跑起一次双环境 eval。
3. 所有路径与命令都与当前实际代码一致。

最终回复必须包含：
- 修改文件列表
- assert CLI
- README 中新增的关键命令
- 一条成功 assert 示例
- 一条失败 assert 示例
```

## 9. 当前推荐执行策略

如果后续要真正开始落代码，建议严格按下面的顺序把 prompts 依次交给 AI：

1. Prompt 1
2. Prompt 2
3. Prompt 3
4. Prompt 4
5. Prompt 5
6. Prompt 7
7. Prompt 8
8. Prompt 6

这里故意把 `Prompt 6: streaming_act full` 放到最后，是因为：

- `act` 和 `streaming_act baseline` 先跑通后，
- 我们就能确定 socket、config、路径、summary、cleanup 这些基础设施没有问题；
- 这样再加 full 模式时，新增问题会集中在在线特征构造层。

## 10. 最终结论

本轮讨论的最终工程结论是：

- **不改 RMBench**
- **保留 RMBench 作为 simulator/evaluator backend**
- **把 bridge、launcher、summary、assert 全部建在 `main/benchmarks/RMBench`**
- **client env 只做 simulator 与轻量 observation 打包**
- **server env 承担全部主仓 checkpoint 加载与 streaming 在线特征构造**

这条路线的优点是：

- 不与 RMBench 原代码结构硬耦合
- 不需要维护 RMBench 内部 patch
- 对 `act` 与 `streaming_act` 都成立
- 后续可继续扩展到更多 `main` 侧 policy
- 能自然接到 `main/outputs/eval/...`、monitor、summary compare、assert 工具链

如果后续严格按上述 prompts 推进，最稳的落地顺序是：

- 先打通 `act`
- 再打通 `streaming_act baseline`
- 最后接 `streaming_act full`

这样风险最低，调试面也最可控。
