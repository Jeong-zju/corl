# 数据转换说明

`data/` 目录里目前有三个常用脚本：

- `convert_dataset.py`：把 ROS `.bag` 转成 LeRobot v3 数据集。
- `download_dataset.py`：从 Hugging Face 下载数据集或其他仓库快照，支持断点续传和自动重试。
- `process_dataset.py`：对已经是 LeRobot 格式的数据集继续做后处理，比如重新切分 episode、重算 signatures。

## 环境准备

在仓库根目录运行：

```bash
cd /home/jeong/zeno/corl/main
conda activate corl-py312
```

转换依赖：

- `opencv-python`
- `pyarrow`
- `rosbags`
- `ffmpeg`
- `ffprobe`
- `huggingface_hub`（下载 Hugging Face 数据集时需要）

如果环境里缺 `rosbags`，安装：

```bash
python -m pip install rosbags
```

## 现在的处理模型

这版脚本不是“先把整条轨迹攒在主进程内存里，再统一保存”。

现在的流程是：

1. 主进程分发 bag 任务，并显示总进度。
2. 每个 worker 进程独立处理一条 bag：
   - 读取并对齐 topic
   - 编码 3 路视频
   - 写单条 episode 的 parquet 到 staging 目录
3. worker 结束后退出；`max_tasks_per_child=1`，进程会被回收，内存也随之释放。
4. 主进程只按顺序合并已经落盘的产物，写最终 `meta/` 信息。

这个设计的目标就是避免“处理速度快于保存速度，结果把多条轨迹同时堆在内存里”。

## 基本用法

按 `/home/jeong/Downloads/Data/convert_hanger_stage3.py` 的默认 topic 和参数转换 `stage3`：

```bash
python data/convert_dataset.py /home/jeong/Downloads/Data/stage3 \
  --dataset-id wholebody_hanger_stage3_v30 \
  --output-dir data/wholebody_hanger_stage3_v30 \
  --task-label pick \
  --fps 30 \
  --image-width 224 \
  --image-height 224 \
  --video-encoder auto \
  --overwrite
```

输出目录示例：

```text
data/wholebody_hanger_stage3_v30
```

## 进度显示

运行时会显示整体 bag 进度条，并实时汇总：

- `saved`
- `skipped`
- `errors`
- `frames`

如果有失败或被跳过的 bag，额外信息会写到：

```text
meta/conversion_issues.log
```

## 默认读取的 topic

```text
/realsense_top/color/image_raw/compressed
/realsense_left/color/image_raw/compressed
/realsense_right/color/image_raw/compressed
/robot/arm_left/joint_states_single
/robot/arm_right/joint_states_single
/teleop/arm_left/joint_states_single
/teleop/arm_right/joint_states_single
/ranger_base_node/odom
```

## 常用参数

- `--bag-glob`：当输入是目录时，控制匹配哪些 `.bag` 文件。
- `--workers`：默认是 `auto`。它会根据当前机器的逻辑 CPU 数和当前 `load average` 自动估算并发数，不需要手动写死。
- `--task-label`：写入任务表和 episode 元数据。
- `--robot-type`：写入 LeRobot 元数据里的机器人类型。
- `--camera-top-topic` / `--camera-left-topic` / `--camera-right-topic`：覆盖相机 topic。
- `--state-left-topic` / `--state-right-topic`：覆盖状态 topic。
- `--action-left-topic` / `--action-right-topic`：覆盖动作 topic。
- `--odom-topic`：覆盖底盘速度 topic；如果不想读取 odom，传 `--odom-topic none`。
- `--video-encoder`：可选 `auto`、`gpu`、`cpu`，或者直接传 ffmpeg encoder 名称，比如 `h264_nvenc`。

## 查看帮助

```bash
python data/convert_dataset.py --help
```

## `download_dataset.py`

`download_dataset.py` 用来从 Hugging Face 拉取仓库快照，默认场景是下载数据集，也可以通过 `--repo-type` 下载 model 或 space。

它的行为有几个点需要注意：

- 支持 `resume_download=True`，网络中断后会基于缓存自动续传。
- 支持失败自动重试，默认 `--max-retries 1000`、`--retry-delay 30`。
- `--max-workers auto` 会根据当前机器 CPU 数和文件描述符上限自动估算下载并发。
- 默认会下载到 `main/data/{dataset_id}`，保留 Hugging Face 的层级目录结构。

### 常见用法

直接下载 `Yinpei/robomme_data_lerobot`，默认会落到 `main/data/Yinpei/robomme_data_lerobot`：

```bash
python data/download_dataset.py Yinpei/robomme_data_lerobot \
  --cache-dir ~/.cache/huggingface
```

输出目录示例：

```text
data/Yinpei/robomme_data_lerobot
```

限制并发并缩短重试等待时间：

```bash
python data/download_dataset.py Yinpei/robomme_data_lerobot \
  --max-workers 8 \
  --max-retries 20 \
  --retry-delay 10
```

### 常用参数

- `--output-dir`：输出根目录；默认就是 `main/data/`，脚本会在里面创建 `<dataset_id>` 对应的层级目录。
- `--cache-dir`：缓存目录；中断后会优先复用这里的缓存续传。
- `--repo-type`：仓库类型，默认是 `dataset`，也可传 `model` 或 `space`。
- `--max-retries`：最大重试次数。
- `--retry-delay`：每次失败后的等待秒数。
- `--max-workers`：下载并发数；默认 `auto`。

### 查看帮助

```bash
python data/download_dataset.py --help
```

## `process_dataset.py`

`process_dataset.py` 用来处理已经存在的 LeRobot 数据集，不负责从 bag 解码。

它支持两类操作：

- `split`：把一条原始 episode 里嵌套的多段轨迹重新切开。
- `update-signatures`：基于状态向量重新计算 `path signature` 和 `delta signature`。

如果两个操作同时传入，会先 `split`，再 `update-signatures`。

### 输入数据

`dataset_id` 可以是：

- 本地数据集路径
- 相对 `main/data/` 的数据集 id，比如 `zeno-ai/wholebody_hanger_stage3_v30`
- Hugging Face 数据集 id；如果本地没有，可以配合 `--download-if-missing`

### 常见用法

只做轨迹切分：

```bash
python data/process_dataset.py zeno-ai/wholebody_hanger_stage3_v30 \
  --operations split \
  --output-dir data/zeno-ai/wholebody_hanger_stage3_v30_split \
  --overwrite-output
```

只更新 signatures：

```bash
python data/process_dataset.py zeno-ai/wholebody_hanger_stage3_v30 \
  --operations update-signatures \
  --output-dir data/zeno-ai/wholebody_hanger_stage3_v30_sig \
  --overwrite-output
```

先切分，再更新 signatures：

```bash
python data/process_dataset.py zeno-ai/wholebody_hanger_stage3_v30 \
  --operations split update-signatures \
  --output-dir data/zeno-ai/wholebody_hanger_stage3_v30_processed \
  --overwrite-output
```

直接原地改写：

```bash
python data/process_dataset.py zeno-ai/wholebody_hanger_stage3_v30 \
  --operations split update-signatures \
  --in-place
```

### 输出行为

- 默认不会改写源数据集，而是输出到一个新目录。
- 如果不传 `--output-dir`，默认输出到 `<原目录名>_processed`。
- 传 `--in-place` 时，会先写到一个临时目录，完成后再覆盖回源数据集。
- 仅做 `split` 时，会同步重写视频切片，使视频和新 episode 对齐。
- 仅做 `update-signatures` 时，不会重新切视频。

### 常用参数

- `--split-strategy auto|done|episode_index`
  - `auto`：按 `next.done`、`frame_index` 回退、`timestamp` 回退来切分
  - `done`：只按 `next.done`
  - `episode_index`：不做嵌套切分
- `--signature-type path|delta|both`：控制重算哪些 signature。
- `--state-key`：指定用于计算 signature 的状态字段，默认是 `observation.state`。
- `--path-signature-window-size`：`0` 表示用完整前缀，`>0` 表示滑动窗口。
- `--path-signature-depth`：signature depth，默认 `3`。
- `--signature-backend auto|simple|signatory`：signature 计算后端。
- `--episodes-per-chunk`：重写后的每个 chunk 放多少个 episode 文件；不传时会优先沿用源数据集配置，如果元数据和实际文件布局不一致，会自动按现有布局推断。
- `--workers`：并行处理 episode 的 worker 数；默认是 `min(8, cpu_count)`。

### 查看帮助

```bash
python data/process_dataset.py --help
```


```bash
./hfd.sh Yinpei/robomme_data_lerobot --dataset --tool aria2c -x 8
```