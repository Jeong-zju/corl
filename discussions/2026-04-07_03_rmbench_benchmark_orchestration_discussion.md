# 2026-04-07 RMBench Benchmark 编排与 `corl-py312` 适配讨论整理

## 1. 讨论背景

本轮讨论的目标是回答一个很实际的工程问题：

- 直接在原始 `RMBench` 目录里运行训练和评估，经常会和 Python 版本、环境依赖、脚本分散等问题纠缠在一起。
- 希望把和虚拟环境、训练入口、评估入口、成功率统计、assert 相关的部分整理到 `main/benchmarks/RMBench` 下。
- 主要目标不是重构整个 benchmark 本体，而是稳定地运行：
  - `act`
  - `streaming_act`
  - 训练
  - eval
  - eval 后输出成功率，并支持阈值断言

这轮讨论的核心不是“是否完全迁移 RMBench 本体”，而是“如何让现有 `main` 侧训练系统和现有 `RMBench` simulator/evaluator 更稳地协作”。

## 2. 本轮确认到的关键事实

### 2.1 `main/benchmarks` 当前是空的

仓库里已经有 `main/benchmarks/` 目录，但目前是空目录。这意味着：

- 很适合把新的 benchmark orchestration 层收口到这里；
- 不需要兼容已有的 benchmark 封装风格；
- 可以直接定义统一的 doctor / train / eval / assert 入口。

### 2.2 `RMBench` 原生已经有两条评估路径

当前 `RMBench` 侧并不是完全没有环境隔离设计，而是有两条比较原始的路径：

1. 单环境 eval：
   - 例如 `RMBench/policy/ACT/eval.sh`
   - 直接在当前环境下运行 `script/eval_policy.py`

2. 双环境 eval：
   - 例如 `RMBench/policy/DP/eval_double_env.sh`
   - server/client 分离
   - 通过 `policy_model_server.py` + `eval_policy_client.py`
   - 支持 `policy_conda_env`

这说明：

- RMBench evaluator 本身并不需要推倒重写；
- 真正缺的是统一编排、标准输出、稳定 CLI 和 `main` checkpoint bridge。

### 2.3 成功率已经会被计算，但输出格式不统一

当前 `RMBench/script/eval_policy.py` 与 `RMBench/script/eval_policy_client.py` 会：

- 计算成功 episode 数
- 打印 success rate
- 写 `_result.txt`
- 写 `eval_log.txt`

但它们还缺少 `main` 风格的稳定评估产物：

- `summary.json`
- 统一字段名
- 阈值断言
- 稳定可发现的输出目录

因此后续工作重点应是：

- 补标准化 summary
- 补 assert 脚本
- 补统一 eval 入口

### 2.4 当前 shell 不是激活态，但 `corl-py312` 实际上已经很完整

本轮排查时确认：

- 当前交互 shell 默认不是 `corl-py312`
- `/usr/bin/python3` 下没有 `torch / lerobot / sapien / mplib`

但进一步检查发现 `corl-py312` 中已经具备本轮工作的关键依赖：

- `torch`
- `torchvision`
- `lerobot`
- `sapien`
- `mplib`
- `h5py`
- `cv2`
- `einops`

这意味着一个非常重要的结论：

- 不应该再默认“必须单独维护一个 RMBench 专属 py310 环境”；
- 优先路线应当是把入口统一成 `conda run -n corl-py312 ...`；
- 只有遇到明确不兼容点时，再退回双环境方案。

### 2.5 `corl-py312` 下 RMBench 基本 smoke check 已经能过

本轮实际验证过：

- `RMBench/policy/ACT` 可在 `corl-py312` 中导入
- `RMBench/script/eval_policy.py` 可导入
- `RMBench/script/eval_policy_client.py` 可导入
- `RMBench/script/policy_model_server.py` 可导入
- `RMBench/script/test_render.py` 中的 `Sapien_TEST()` 可运行并输出 `Render Well`

这说明：

- `corl-py312` 不只是理论可用；
- 至少对当前 RMBench 的导入、渲染探针、基础 evaluator 启动链路已经具备可行性。

### 2.6 RMBench 数据转 LeRobot 数据集的链路其实已经存在

本轮还确认了一个非常关键的仓库现状：

- `main/data/convert_dataset.py` 已经支持 `--convert rmbench`
- `main/data/rmbench/collection.json` 已存在
- `main/data/rmbench/<task>` 已经是标准 LeRobot v3.0 风格本地数据集
- 这些数据集能被 `LeRobotDataset` 直接读取

这意味着：

- “把 RMBench 数据接到 `main/scripts/train_policy.py`” 不是从零开始；
- 数据层已经基本具备；
- 更值得投入的是训练 defaults、bridge、eval orchestration。

## 3. 当前讨论收敛出的核心结论

### 3.1 不建议物理搬迁 RMBench 核心代码

本轮最重要的设计结论是：

- 不建议把 `RMBench/envs`、`RMBench/script`、`RMBench/task_config` 等核心 benchmark backend 物理搬到 `main/benchmarks/RMBench`

原因：

- `RMBench` 现有相对路径、import 结构、task_config 依赖较重；
- 物理搬迁会引入大量无收益的路径回归问题；
- 当前问题的本质并不是目录位置，而是环境编排和接口标准化。

因此更稳的方案是：

- **保留 `RMBench` 作为 backend**
- **把 orchestration 层整理到 `main/benchmarks/RMBench`**

### 3.2 真正应迁移的是“虚拟环境相关”和“编排相关”的部分

建议放到 `main/benchmarks/RMBench` 的内容包括：

- 环境解析与 doctor 脚本
- `conda run -n corl-py312` 统一入口
- 统一 train.sh
- 统一 eval.sh
- success rate summary 生成
- success rate assert
- `main` checkpoint 到 RMBench evaluator 的 bridge
- README 和 smoke tests

### 3.3 默认应优先走单环境 `corl-py312`

既然本轮已经确认：

- `corl-py312` 中有 `lerobot`
- `corl-py312` 中有 `sapien / mplib`
- `Sapien_TEST()` 可过

那么默认设计应为：

- 训练：`conda run -n corl-py312 python main/scripts/train_policy.py ...`
- 评估：`conda run -n corl-py312 python RMBench/script/eval_policy.py ...`

只有以下情况才考虑双环境：

- 某个 policy 自身有强绑定的旧版本环境
- 某个外部模型依赖无法在 `corl-py312` 共存
- 某个 server/client policy 已经天然采用双环境部署

### 3.4 `streaming_act` 应先跑 baseline，再扩展 full 模式

本轮还明确了一个实现优先级：

`streaming_act` 不应一上来就把所有 signature / prefix memory / delta signature / online memory 一次性接到 RMBench。

更稳的顺序是：

1. `ACT`
2. `streaming_act baseline`
3. `streaming_act full`

其中 `streaming_act baseline` 建议先关闭：

- `use_path_signature`
- `use_delta_signature`
- `use_prefix_sequence_training`
- `use_visual_prefix_memory`

先证明训练、checkpoint 加载、RMBench rollout、summary 产出都能闭环，再扩 full 模式。

## 4. 推荐的总体架构

建议形成如下分层：

### 4.1 backend 层

保留现状：

- `RMBench/envs`
- `RMBench/script`
- `RMBench/task_config`
- `RMBench/assets`
- `RMBench/data`

### 4.2 main 训练层

继续使用现有：

- `main/scripts/train_policy.py`
- `main/scripts/eval_policy.py`
- `main/bash/train_policy.sh`
- `main/bash/eval_policy.sh`

### 4.3 RMBench orchestration 层

新增：

- `main/benchmarks/RMBench/README.md`
- `main/benchmarks/RMBench/scripts/common.sh`
- `main/benchmarks/RMBench/scripts/doctor.sh`
- `main/benchmarks/RMBench/scripts/train.sh`
- `main/benchmarks/RMBench/scripts/eval.sh`
- `main/benchmarks/RMBench/scripts/prepare_streaming_dataset.sh`
- `main/benchmarks/RMBench/scripts/smoke.sh`
- `main/benchmarks/RMBench/python/rmbench_bridge/...`
- `main/benchmarks/RMBench/python/assert_eval_summary.py`

### 4.4 bridge 层

bridge 的职责不是训练，而是：

- 把 `main` 训练出来的 checkpoint 接到 RMBench evaluator
- 把 RMBench observation 转成 LeRobot policy 输入
- 把 eval 输出整理成标准 summary

至少应有：

- `lerobot_act.py`
- `lerobot_streaming_act.py`

## 5. 推荐落地顺序

建议按下面顺序推进：

1. 先做 doctor 和统一环境入口
2. 再补 RMBench task 的训练 defaults
3. 先打通 `ACT` 的训练与 eval bridge
4. 再打通 `streaming_act baseline`
5. 最后再接 full `streaming_act`

这样可以把风险分层隔离：

- 环境风险
- 路径风险
- checkpoint bridge 风险
- online memory 风险

## 6. 面向 AI Coding 的分步 prompts

下面是本轮讨论最终整理出的分步 AI Coding prompts。设计原则是：

- 每一步单独可执行
- 每一步都有明确边界
- 每一步都优先保持向后兼容
- 每一步都尽量先打通最小闭环

### Prompt 1：搭建 `main/benchmarks/RMBench` 骨架与 doctor

```text
你在 /home/jeong/zeno/corl 仓库中工作。请不要物理移动 /home/jeong/zeno/corl/RMBench 的核心代码；把“编排层”整理到 /home/jeong/zeno/corl/main/benchmarks/RMBench 下。

任务:
1. 创建目录:
   - main/benchmarks/RMBench/README.md
   - main/benchmarks/RMBench/scripts/common.sh
   - main/benchmarks/RMBench/scripts/doctor.sh
   - main/benchmarks/RMBench/python/__init__.py
2. common.sh 负责统一解析:
   - REPO_ROOT
   - MAIN_ROOT
   - RMBENCH_ROOT
   - 默认 conda env: corl-py312
3. doctor.sh 必须使用 conda run -n corl-py312，不要假设当前 shell 已激活环境。
4. doctor.sh 至少检查:
   - conda env corl-py312 是否存在
   - python imports: torch, lerobot, sapien, mplib, h5py, cv2
   - ffmpeg / ffprobe 是否存在
   - main/data/rmbench/collection.json 是否存在
   - 能否在 corl-py312 下运行 RMBench/script/test_render.py 中的 Sapien_TEST
5. README 先写清楚设计原则:
   - RMBench 保留为 backend
   - main/benchmarks/RMBench 只负责 orchestration
   - 所有入口默认走 conda run -n corl-py312

要求:
- 脚本使用 bash 且 set -euo pipefail
- 不要改动任何训练/评估核心逻辑
- 输出清晰的 pass/fail 信息

完成后请汇报:
- 新增文件
- doctor.sh 的检查项
- 后续步骤建议
```

### Prompt 2：补 RMBench task 的训练 defaults 与统一 train 入口

```text
继续在 /home/jeong/zeno/corl 工作。目标是让 RMBench task 能通过 main/scripts/train_policy.py 直接训练。

任务:
1. 阅读:
   - main/data/rmbench/collection.json
   - main/bash/train_policy.sh
   - main/scripts/policy_defaults.py
2. 为 RMBench task 增加 defaults。优先方案:
   - main/bash/defaults/rmbench/<task>/act.yaml
   - main/bash/defaults/rmbench/<task>/streaming_act.yaml
3. dataset_root 统一指向:
   - data/rmbench/<task>
4. output_root 建议统一到:
   - outputs/train/rmbench/<task>/act
   - outputs/train/rmbench/<task>/streaming_act
5. streaming_act 默认先走 baseline:
   - use_path_signature: false
   - use_delta_signature: false
   - use_prefix_sequence_training: false
   - use_visual_prefix_memory: false
   - n_action_steps: 1 或一个明显安全的值，并解释原因
6. 新增统一训练脚本:
   - main/benchmarks/RMBench/scripts/train.sh
7. train.sh 支持:
   - --task <task_name>
   - --policy act|streaming_act
   - --dry-run
   - 透传额外训练参数到 main/scripts/train_policy.py
8. train.sh 必须通过 conda run -n corl-py312 调用训练，不假设激活环境。

要求:
- 尽量自动从 collection.json 发现 task，而不是手写死
- 如果 defaults 数量较多，可以写生成脚本，但最终仓库里要有稳定可用的 yaml 产物
- 不要引入新的训练框架

验收:
- bash main/benchmarks/RMBench/scripts/train.sh --task cover_blocks --policy act --dry-run
- bash main/benchmarks/RMBench/scripts/train.sh --task cover_blocks --policy streaming_act --dry-run

完成后请汇报:
- 生成了哪些 defaults
- baseline streaming_act 的关键开关
- 实际训练命令示例
```

### Prompt 3：先打通 `ACT` 的 RMBench eval bridge

```text
继续在 /home/jeong/zeno/corl 工作。现在实现 ACT 的 eval bridge：训练仍用 main/scripts/train_policy.py，评估仍用 RMBench simulator，但 policy 加载改为加载 main 的 LeRobot ACT checkpoint。

任务:
1. 新增 bridge package:
   - main/benchmarks/RMBench/python/rmbench_bridge/__init__.py
   - main/benchmarks/RMBench/python/rmbench_bridge/lerobot_act.py
2. lerobot_act.py 必须暴露与 RMBench evaluator 兼容的接口:
   - get_model(usr_args)
   - eval(TASK_ENV, model, observation)
   - reset_model(model)
3. 该 bridge 负责:
   - 从 main 训练产物目录加载 ACT checkpoint
   - 把 RMBench observation 映射到 LeRobot policy 输入
   - 三路图像键使用:
     - observation.images.cam_high
     - observation.images.cam_left_wrist
     - observation.images.cam_right_wrist
   - 状态使用 14 维 joint state
4. 新增 bridge config:
   - main/benchmarks/RMBench/configs/policies/act_bridge.yml
5. 不要要求把 bridge 文件放进 RMBench/policy；优先通过 PYTHONPATH + policy_name=rmbench_bridge.lerobot_act 的方式接入 importlib。
6. 必要时小幅修改 RMBench/script/eval_policy.py，但必须保持向后兼容。至少补这几个能力:
   - 可配置 test_num / num_episodes，而不是硬编码 100
   - 可配置 output_root 或稳定可发现输出目录
   - 额外写出 summary.json，字段至少包含:
     task_name, policy_name, success_count, num_episodes, success_rate, seed, result_txt_path
7. 新增统一 eval 入口:
   - main/benchmarks/RMBench/scripts/eval.sh
8. eval.sh 至少支持:
   - --task
   - --policy act
   - --policy-path <main checkpoint dir>
   - --task-config demo_clean
   - --num-episodes
   - --seed
   - --gpu-id
   - --output-dir
   - --min-success-rate
9. eval.sh 结束后如果传了 --min-success-rate，则根据 summary.json assert 退出码。

要求:
- 默认环境仍然是 corl-py312
- 尽量少改 RMBench 本体，优先把新逻辑放在 main/benchmarks/RMBench
- summary.json 路径要稳定，方便后续 monitor/assert

完成后请汇报:
- bridge 的输入输出映射
- 对 RMBench evaluator 做了哪些最小修改
- 一个可执行的 ACT eval 命令
```

### Prompt 4：再打通 `streaming_act baseline`

```text
继续在 /home/jeong/zeno/corl 工作。现在实现 streaming_act baseline 的 RMBench eval bridge。目标是先跑通 end-to-end，不启用 path signature / prefix memory。

任务:
1. 新增:
   - main/benchmarks/RMBench/python/rmbench_bridge/lerobot_streaming_act.py
   - main/benchmarks/RMBench/configs/policies/streaming_act_bridge.yml
2. 接口同 ACT bridge:
   - get_model(usr_args)
   - eval(TASK_ENV, model, observation)
   - reset_model(model)
3. 必须从 main/policy/lerobot_policy_streaming_act 和 main 训练产物加载 checkpoint。
4. baseline 模式要求:
   - use_path_signature = false
   - use_delta_signature = false
   - use_prefix_sequence_training = false
   - use_visual_prefix_memory = false
5. eval 时优先使用 n_action_steps=1，避免流式缓存导致动作延迟问题。
6. 复用统一的 main/benchmarks/RMBench/scripts/eval.sh，通过 --policy streaming_act 路由到这个 bridge。
7. 如果 checkpoint 配置要求额外特征而当前 wrapper 没提供，报清晰错误，不要 silent fail。

要求:
- 不要先实现 full SIPM 模式
- 先保证 baseline 简单稳定
- 尽量复用 ACT bridge 的公共代码

验收:
- eval.sh 能根据 --policy streaming_act 正确加载 bridge
- 能写出 summary.json
- --min-success-rate assert 生效

完成后请汇报:
- 与 ACT bridge 共用了哪些逻辑
- baseline streaming_act 在 RMBench 中的输入结构
- 下一步如何扩展到 full 模式
```

### Prompt 5：补 full `streaming_act` 的数据准备与可选增强模式

```text
继续在 /home/jeong/zeno/corl 工作。现在补 RMBench -> full streaming_act 的增强模式，但要保持 baseline 仍然可用。

任务:
1. 新增:
   - main/benchmarks/RMBench/scripts/prepare_streaming_dataset.sh
   - main/benchmarks/RMBench/python/assert_eval_summary.py
2. prepare_streaming_dataset.sh 调用:
   - main/data/process_dataset.py
3. 目标是从:
   - main/data/rmbench/<task>
   生成一个适合 full streaming_act 的处理后数据集，例如:
   - main/data/rmbench_processed/<task>
4. 先只要求 update-signatures；不要强行做额外重切分，除非代码证明必须。
5. 为 full 模式补一套 defaults 或通过 train.sh 的 flag 切换:
   - --streaming-mode baseline|full
6. full 模式下:
   - train 使用 processed dataset
   - 如果 checkpoint config 需要 path signature，则 eval bridge 读取 config 后在线构造 observation.path_signature
   - 如果实现成本合适，再支持 delta signature
   - 如果 prefix memory/full 模式当前还不能稳定支持，请明确 gate 掉，并给出清晰错误
7. assert_eval_summary.py 负责:
   - 读取 summary.json
   - 校验 success_rate 是否达到阈值
   - 失败时打印清晰报错并返回非 0

要求:
- baseline 模式绝对不能被 full 模式破坏
- full 模式优先支持 path signature；prefix memory 若实现复杂，可以先明确限制
- 文档中写清 baseline 与 full 的区别

完成后请汇报:
- processed dataset 路径规范
- full 模式目前支持到哪一步
- assert 脚本的用法
```

### Prompt 6：收尾 README 与 smoke test

```text
继续在 /home/jeong/zeno/corl 工作。现在把 RMBench 编排层收尾。

任务:
1. 新增:
   - main/benchmarks/RMBench/scripts/smoke.sh
2. smoke.sh 至少覆盖:
   - doctor
   - ACT train dry-run
   - streaming_act train dry-run
   - ACT eval dry-run 或最小 episode 数 smoke
   - streaming_act eval dry-run 或最小 episode 数 smoke
3. README.md 需要给出一条最短路径:
   - 环境自检
   - 如果 raw RMBench 数据变了，如何重新 convert
   - ACT 训练
   - streaming_act baseline 训练
   - ACT eval + success_rate assert
   - streaming_act eval + success_rate assert
   - full 模式如何额外处理数据
4. README 中明确说明:
   - 当前默认环境是 corl-py312
   - 当前 shell 不需要先手动 conda activate，脚本内部会用 conda run
   - RMBench 本体保留为 backend，不建议搬核心目录
5. 如果你在实现过程中对 RMBench/script/eval_policy.py 或 eval_policy_client.py 做了兼容性修改，请在 README 中列出这些新增 CLI 参数。

要求:
- 命令示例全部使用仓库内真实路径和真实脚本名
- 输出尽量统一到 main/outputs/train 和 main/outputs/eval
- 最终文档要足够让别人照抄命令

完成后请汇报:
- 最终命令矩阵
- 已知限制
- 建议先跑的 smoke task
```

## 7. 本轮建议的实施顺序

为了降低复杂度，推荐严格按以下顺序执行：

1. Prompt 1
2. Prompt 2
3. Prompt 3
4. Prompt 4
5. Prompt 5
6. Prompt 6

如果资源有限，优先级可进一步压缩为：

1. 先做 doctor
2. 先做 ACT train/eval 闭环
3. 再做 `streaming_act baseline`
4. 最后再考虑 `streaming_act full`

## 8. 本轮最终结论

本轮讨论最终收敛出的结论可以概括为四点：

1. **可以整理到 `main/benchmarks/RMBench`，但不应物理搬迁 RMBench backend。**
2. **默认环境应优先统一到 `corl-py312`。**
3. **训练尽量继续复用 `main`，评估继续复用 RMBench simulator，但中间加 bridge。**
4. **`streaming_act` 应先做 baseline，再做 full。**

换句话说，本轮真正推荐的不是“迁移 benchmark 本体”，而是：

**把 RMBench 变成 `main` 体系下一个由 `main/benchmarks/RMBench` 统一编排的 backend benchmark。**

