# Eval 接口规范

下面的命令默认都在当前工作目录执行。

统一通过下面这个入口做评估：

```bash
./bash/eval_policy.sh ...
```

它最终会调用 `scripts/eval_policy.py`，并把不同数据集 / 环境的参数先做一轮标准化，再分发到具体 evaluator。

## 1. 两种评估模式

### 1.1 离线数据集评估

适用于 held-out split / replay-style eval：

```bash
./bash/eval_policy.sh \
  --policy act \
  --dataset zeno-ai/wholebody_hanger_stage3_v30 \
  --policy-path <ckpt_dir>
```

核心参数：

- `--dataset`: 统一的数据集选择器。支持 repo id、`data/...`、相对路径。
- `--policy`
- `--policy-path` 或 `--latest-run-dir`
- `--eval-split {train,test}`
- `--max-episodes`
- `--max-steps-per-episode`
- `--test-ratio`
- `--split-seed`
- `--shuffle-split-episodes` / `--preserve-split-order`

### 1.2 在线环境 rollout 评估

适用于 simulator rollout / success-rate eval：

```bash
./bash/eval_policy.sh \
  --env metaworld \
  --policy streaming_act
```

核心参数：

- `--env`
- `--policy`
- `--policy-path` 或 `--latest-run-dir`
- `--task` 或 `--tasks`
- `--num-rollouts`
- `--max-steps`
- `--fps`
- `--max-episodes-rendered`

## 1.3 最常用的几种调用

下面这组命令适合作为快速参考：

```bash
# 1) 用 defaults 对应的最新训练结果做 Meta-World rollout eval
./bash/eval_policy.sh --env metaworld --policy act

# 2) 用显式 checkpoint 做离线 held-out eval
./bash/eval_policy.sh \
  --policy streaming_act \
  --dataset zeno-ai/wholebody_hanger_stage3_v30 \
  --policy-path <ckpt_dir>

# 3) 做 RoboCasa 单任务 rollout eval，task 会从 dataset 自动推断
./bash/eval_policy.sh \
  --env robocasa \
  --policy act \
  --dataset robocasa/composite/ArrangeBreadBasket \
  --policy-path <ckpt_dir>

# 4) 做 RoboCasa diffusion 单任务 rollout eval
./bash/eval_policy.sh \
  --env robocasa \
  --policy diffusion \
  --dataset robocasa/atomic/CloseFridge \
  --policy-path <ckpt_dir>

# 5) 做 RoboCasa 多任务 rollout eval
./bash/eval_policy.sh \
  --env robocasa \
  --policy act \
  --tasks ArrangeBreadBasket,PickPlaceCounterToSink \
  --policy-path <ckpt_dir>

# 6) 覆盖 rollout 数、步数和输出目录
./bash/eval_policy.sh \
  --env metaworld \
  --policy streaming_act \
  --num-rollouts 10 \
  --max-steps 300 \
  --output-dir outputs/eval/manual/metaworld-streaming-act
```

## 2. 标准化规则

### 2.1 `--task` / `--tasks`

- `--tasks` 是 `--task` 的别名。
- 传入逗号分隔列表时，会自动：
  - 去掉首尾空格
  - 去重
  - 规范成稳定顺序的 `a,b,c`
- 规范化后的结果会统一写回 `args.task`，环境 evaluator 只消费这一套语义。

例子：

```bash
--task "ArrangeBreadBasket , ArrangeBreadBasket, PickPlaceCounterToSink"
```

会被标准化为：

```text
ArrangeBreadBasket,PickPlaceCounterToSink
```

### 2.2 RoboCasa task 解析优先级

`--env robocasa` 时，如果没显式传 `--task/--tasks`，会按下面顺序自动补全：

1. 显式 `--task` / `--tasks`
2. defaults 里的 `dataset_tasks`
3. `--dataset` / `--dataset-repo-id` / defaults `dataset_root` 的叶子 task 名

所以这两种写法现在都合法：

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy act \
  --dataset robocasa/composite/ArrangeBreadBasket
```

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy streaming_act \
  --dataset robocasa/composite
```

第二种会从 defaults 里的 `dataset_tasks` 自动解析出 `ArrangeBreadBasket`。

### 2.3 RoboCasa 专属参数

统一保留两项 RoboCasa 扩展参数：

- `--robocasa-conda-env`
- `--robocasa-split {target,pretrain,all}`

推荐默认值：

- `robocasa_conda_env=robocasa`
- `robocasa_split=target`

### 2.4 公共 rollout 参数

所有在线环境统一使用下面这组字段：

- `--num-rollouts`: rollout 数
- `--max-steps`: 单条 rollout 最大步数
- `--fps`: 保存视频帧率
- `--max-episodes-rendered`: 每个 task 最多导出多少条视频

内部会被归一成：

- `args.eval_num_rollouts`
- `args.eval_max_steps`
- `args.eval_fps`
- `args.eval_max_episodes_rendered`
- `args.eval_seed`
- `args.eval_task_spec`
- `args.eval_task_names`

## 3. RoboCasa 推荐用法

### 3.1 单任务数据集，省略 `--task`

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy act \
  --dataset robocasa/composite/ArrangeBreadBasket \
  --policy-path <ckpt_dir>
```

### 3.2 单任务 diffusion checkpoint

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy diffusion \
  --dataset robocasa/atomic/CloseFridge \
  --policy-path <ckpt_dir>
```

### 3.3 collection defaults 自动补 task

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy streaming_act \
  --dataset robocasa/atomic \
  --policy-path <ckpt_dir>
```

### 3.4 显式多任务

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy act \
  --tasks ArrangeBreadBasket,PickPlaceCounterToSink \
  --num-rollouts 10 \
  --max-steps 250 \
  --policy-path <ckpt_dir>
```

### 3.5 指定 RoboCasa split

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy act \
  --dataset robocasa/composite/ArrangeBreadBasket \
  --robocasa-split all \
  --policy-path <ckpt_dir>
```

## 4. 示例库

这一节按“直接能复制”的方式列出常见调用。

### 4.1 离线数据集评估

#### ACT / Streaming ACT 基础用法

```bash
./bash/eval_policy.sh \
  --policy act \
  --dataset zeno-ai/wholebody_hanger_stage3_v30 \
  --policy-path <ckpt_dir>

./bash/eval_policy.sh \
  --policy streaming_act \
  --dataset zeno-ai/wholebody_hanger_stage3_v30 \
  --policy-path <ckpt_dir>
```

#### Diffusion / PRISM Diffusion

```bash
./bash/eval_policy.sh \
  --policy diffusion \
  --dataset zeno-ai/day3_5_Exp1_processed \
  --policy-path <ckpt_dir>

./bash/eval_policy.sh \
  --policy prism_diffusion \
  --dataset robocasa/atomic/CloseFridge \
  --policy-path <ckpt_dir>
```

#### 不传 `--policy-path`，直接使用默认训练目录里的最新 run

```bash
./bash/eval_policy.sh \
  --policy act \
  --dataset zeno-ai/wholebody_hanger_stage3_v30

./bash/eval_policy.sh \
  --policy streaming_act \
  --dataset robocasa/composite
```

#### 显式指定 latest run 目录

```bash
./bash/eval_policy.sh \
  --policy act \
  --dataset zeno-ai/wholebody_hanger_stage3_v30 \
  --latest-run-dir outputs/train/zeno-ai/wholebody_hanger_stage3_v30/act/20260417_120000
```

#### 评估 train split

```bash
./bash/eval_policy.sh \
  --policy act \
  --dataset zeno-ai/wholebody_hanger_stage3_v30 \
  --policy-path <ckpt_dir> \
  --eval-split train
```

#### 限制评估 episode 数和每条 episode 的步数

```bash
./bash/eval_policy.sh \
  --policy streaming_act \
  --dataset zeno-ai/wholebody_hanger_stage3_v30 \
  --policy-path <ckpt_dir> \
  --max-episodes 8 \
  --max-steps-per-episode 200
```

#### 用 `test_ratio + split_seed` 重建 split

```bash
./bash/eval_policy.sh \
  --policy act \
  --dataset robocasa/composite/ArrangeBreadBasket \
  --policy-path <ckpt_dir> \
  --test-ratio 0.2 \
  --split-seed 42 \
  --shuffle-split-episodes
```

#### 覆盖 `n_action_steps`

```bash
./bash/eval_policy.sh \
  --policy streaming_act \
  --dataset robocasa/composite/ArrangeBreadBasket \
  --policy-path <ckpt_dir> \
  --n-action-steps 1
```

#### 显式指定 signature backend

```bash
./bash/eval_policy.sh \
  --policy streaming_act \
  --dataset robocasa/composite/ArrangeBreadBasket \
  --policy-path <ckpt_dir> \
  --signature-backend simple
```

### 4.2 Meta-World rollout eval

#### 默认 task 子集

```bash
./bash/eval_policy.sh --env metaworld --policy act
./bash/eval_policy.sh --env metaworld --policy streaming_act
```

#### 显式指定 task 子集

```bash
./bash/eval_policy.sh \
  --env metaworld \
  --policy act \
  --task assembly-v3,dial-turn-v3,handle-press-side-v3
```

#### 覆盖 rollout 数、步数、视频数量

```bash
./bash/eval_policy.sh \
  --env metaworld \
  --policy streaming_act \
  --num-rollouts 20 \
  --max-steps 300 \
  --max-episodes-rendered 8
```

#### 显式指定 checkpoint 和输出目录

```bash
./bash/eval_policy.sh \
  --env metaworld \
  --policy act \
  --policy-path <ckpt_dir> \
  --output-dir outputs/eval/manual/metaworld-act
```

### 4.3 h_shape rollout eval

#### 默认用法

```bash
./bash/eval_policy.sh \
  --env h_shape \
  --policy act \
  --policy-path <ckpt_dir>
```

#### 调整成功阈值和动作裁剪

```bash
./bash/eval_policy.sh \
  --env h_shape \
  --policy streaming_act \
  --policy-path <ckpt_dir> \
  --num-rollouts 50 \
  --max-steps 120 \
  --success-threshold 0.08 \
  --max-action-step 0.10
```

### 4.4 braidedhub rollout eval

#### 默认用法

```bash
./bash/eval_policy.sh \
  --env braidedhub \
  --policy act \
  --policy-path <ckpt_dir>
```

#### 随机起点 + detect 模式

```bash
./bash/eval_policy.sh \
  --env braidedhub \
  --policy streaming_act \
  --policy-path <ckpt_dir> \
  --enable-randomize \
  --collision-mode detect \
  --num-rollouts 40 \
  --max-action-step 0.10
```

### 4.5 panda_route rollout eval

#### 默认用法

```bash
./bash/eval_policy.sh \
  --env panda_route \
  --policy act \
  --policy-path <ckpt_dir>
```

#### 小步长重规划

```bash
./bash/eval_policy.sh \
  --env panda_route \
  --policy streaming_act \
  --policy-path <ckpt_dir> \
  --n-action-steps 1 \
  --max-action-step 0.05 \
  --num-rollouts 30
```

### 4.6 RoboCasa rollout eval

#### 单任务，task 从 dataset 叶子名自动推断

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy act \
  --dataset robocasa/composite/ArrangeBreadBasket \
  --policy-path <ckpt_dir>

./bash/eval_policy.sh \
  --env robocasa \
  --policy streaming_act \
  --dataset robocasa/atomic/CloseFridge \
  --policy-path <ckpt_dir>
```

#### collection 级 defaults 自动补 task

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy act \
  --dataset robocasa/composite \
  --policy-path <ckpt_dir>

./bash/eval_policy.sh \
  --env robocasa \
  --policy streaming_act \
  --dataset robocasa/atomic \
  --policy-path <ckpt_dir>
```

#### 显式多任务

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy act \
  --tasks ArrangeBreadBasket,PickPlaceCounterToSink \
  --policy-path <ckpt_dir> \
  --num-rollouts 10 \
  --max-steps 250
```

#### 显式切换 split

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy act \
  --dataset robocasa/composite/ArrangeBreadBasket \
  --robocasa-split target \
  --policy-path <ckpt_dir>

./bash/eval_policy.sh \
  --env robocasa \
  --policy act \
  --dataset robocasa/composite/ArrangeBreadBasket \
  --robocasa-split all \
  --policy-path <ckpt_dir>
```

#### 指定 RoboCasa conda 环境

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy streaming_act \
  --dataset robocasa/atomic/CloseFridge \
  --policy-path <ckpt_dir> \
  --robocasa-conda-env robocasa
```

#### 强制覆盖 `max_steps`

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy act \
  --dataset robocasa/composite/ArrangeBreadBasket \
  --policy-path <ckpt_dir> \
  --max-steps 300
```

#### 限制视频导出数量

```bash
./bash/eval_policy.sh \
  --env robocasa \
  --policy streaming_act \
  --dataset robocasa/composite/ArrangeBreadBasket \
  --policy-path <ckpt_dir> \
  --num-rollouts 20 \
  --max-episodes-rendered 4
```

### 4.7 统一覆盖输出目录和 run tag

#### 自定义输出目录

```bash
./bash/eval_policy.sh \
  --env metaworld \
  --policy act \
  --output-dir outputs/eval/manual/metaworld-act
```

#### 使用带 `{run_tag}` 的路径模板

```bash
./bash/eval_policy.sh \
  --policy act \
  --dataset zeno-ai/wholebody_hanger_stage3_v30 \
  --policy-path <ckpt_dir> \
  --output-dir outputs/eval/custom/zeno-act/{run_tag} \
  --run-tag smoke_test
```

### 4.8 直接使用 Python 入口

当你想绕过 shell wrapper 时，也可以直接调用：

```bash
python3 scripts/eval_policy.py \
  --env metaworld \
  --policy act \
  --policy-path <ckpt_dir>

python3 scripts/eval_policy.py \
  --policy streaming_act \
  --dataset robocasa/composite/ArrangeBreadBasket \
  --policy-path <ckpt_dir>
```

## 5. 结果目录

评估输出统一写到：

```text
outputs/eval/...
```

常见产物：

- `summary.json`
- `eval_info.json`（例如 Meta-World）
- `videos/` 或 rollout mp4

## 6. 当前约束

- RoboCasa 在线 env eval 目前支持 `act`、`diffusion` 和 `streaming_act`
- RoboCasa 的 `prism_diffusion` 仍然只支持 dataset eval
- `task` 只是 env rollout 语义，dataset eval 会忽略它
