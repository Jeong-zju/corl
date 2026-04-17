# 数据目录说明

`main/data/` 里现在主要维护五个工具：

- `convert_legacy_dataset_to_v30.py`：把 legacy 的 LeRobot v2.1 数据集转换成 v3.0 数据集，并带进度条展示转换过程。
- `convert_rosbag_to_lerobot.py`：通过 YAML 配置把 ROS bag 转成 LeRobot v3.0 数据集，适合像 hanger 这样需要自定义图像 key 映射和 state/action 拼接规则的数据。
- `process_dataset.py`：为已有的 LeRobot 数据集计算全前缀 `path signature` / `delta signature`，把结果写到独立的 signature cache，并同步更新数据集元信息。
- `hfd.sh`：从 Hugging Face 下载 model 或 dataset 仓库快照，适合拉取原始数据集到本地。
- `upload_dataset_to_hf.py`：把本地数据集目录上传到 Hugging Face dataset 仓库，支持自动建仓、路径过滤，以及大目录的可恢复上传。

## 环境准备

在仓库根目录运行：

```bash
cd /path/to/project-root
conda activate corl-py312
```

`process_dataset.py` 依赖：

- `numpy`
- `pyarrow`
- `torch`
- `signatory`
- `tqdm`（可选，仅用于进度条）

## `convert_legacy_dataset_to_v30.py`

这个脚本用于把 legacy 的 LeRobot `v2.1` 数据集转换成 `v3.0` 数据集。

它会：

1. 识别输入数据集是否还是 legacy `v2.1`
2. 调用 LeRobot 官方 `v2.1 -> v3.0` 转换流程
3. 用 `tqdm` 展示扫描和阶段进度
4. 默认把额外产物一并复制到新数据集目录里，例如 `.signature_cache`

### 常见用法

把 `ArrangeBreadBasket` 转成同级的 `ArrangeBreadBasket_v30`：

```bash
python data/convert_legacy_dataset_to_v30.py robocasa/composite/ArrangeBreadBasket
```

显式指定输出目录：

```bash
python data/convert_legacy_dataset_to_v30.py \
  --dataset robocasa/composite/ArrangeBreadBasket \
  --output-dir data/robocasa/composite/ArrangeBreadBasket_v30
```

允许覆盖已有输出目录：

```bash
python data/convert_legacy_dataset_to_v30.py \
  robocasa/composite/ArrangeBreadBasket \
  --overwrite-output
```

如果不想复制 `.signature_cache` 这类额外文件：

```bash
python data/convert_legacy_dataset_to_v30.py \
  robocasa/composite/ArrangeBreadBasket \
  --skip-extra-artifacts
```

### 注意事项

- 该脚本不会原地覆盖 legacy 数据集；输出必须是一个新目录。
- 运行时依赖 `lerobot` 官方转换模块；请先进入正确环境，比如 `conda activate corl-py312`。
- 如果安装了 `tqdm`，终端里会看到扫描进度条和转换阶段进度条。

如果要用 `hfd.sh`，还需要：

- `curl`
- `aria2c` 或 `wget`
- `jq`（可选；有的话鉴权检查更稳）

如果要用 `upload_dataset_to_hf.py`，还需要：

- `huggingface-hub`
- 已执行 `huggingface-cli login` / `hf auth login`，或者提供 `HF_TOKEN` / `--token`

## `convert_rosbag_to_lerobot.py`

这个脚本用于把 ROS bag 直接转换成 LeRobot `v3.0` 数据集，并把下面这些 dataset-specific 逻辑放进 YAML：

- 图像话题到 `observation.images.*` key 的映射
- 各个状态量/动作量从 ROS message 里怎么提取
- `observation.state`、`action` 等向量特征由哪些源按什么顺序拼接
- 相对项目根目录的输入/输出路径、FPS、图像尺寸、视频编码、并行 worker 数等参数

脚本当前采用“两遍读取 + 增量写入”的方式控制内存占用：

- 第一遍只扫描各 topic 的时间戳，规划每一帧应采样的消息索引
- 第二遍只反序列化真正会被用到的消息，并直接把 frame 逐条写入 LeRobot 数据集
- `processing.n_workers` 仍会保留在配置和摘要里，但转换阶段现在会按 bag 顺序处理，以避免多个 worker 同时缓存整包数据导致 OOM

当前仓库附带了一份 zeno 示例配置：

```bash
python data/convert_rosbag_to_lerobot.py \
  --config data/configs/zeno_conversion.yaml \
  --validate-config
```

实际转换时去掉 `--validate-config` 即可：

```bash
python data/convert_rosbag_to_lerobot.py \
  --config data/configs/zeno_conversion.yaml
```

如果输出目录已存在，也可以临时覆盖配置里的行为：

```bash
python data/convert_rosbag_to_lerobot.py \
  --config data/configs/zeno_conversion.yaml \
  --overwrite-output
```

配置里的 `dataset.data_root` 和 `dataset.output_dir` 现在都必须写成相对项目根目录的路径，例如 `data/WholeBody_Hanger/stage3`。
脚本会自动通过自身所在位置推导项目根目录，所以即使 `main` 目录改名，也不需要改 YAML 里的前缀。

## `process_dataset.py`

这个脚本现在只做一件事：为 LeRobot 数据集构建 signature cache。

它不会再执行 README 旧版本里提到的 `split`、`merge`、`update-signatures` 多阶段处理；当前版本的职责是：

1. 读取 `data/chunk-*/*.parquet`
2. 按 `episode_index` 和 `frame_index` 还原每条 episode 的时序
3. 基于 `observation.state`（默认）计算每一帧的 full-prefix path signature
4. 由 path signature 进一步计算 delta signature
5. 把结果写入 `.npy` cache 文件，而不是回写到 parquet 列里
6. 更新 `meta/info.json` 和 `meta/stats.json`
7. 如果 parquet 里本来已有 signature 列，会把这些列移除

### 输入要求

输入目录需要是一个可识别的 LeRobot 数据集根目录，至少包含：

- `meta/info.json`
- `meta/stats.json`
- `data/chunk-*/*.parquet`

脚本还会读取 parquet 中这些列：

- `index`
- `episode_index`
- `observation.state`（或你通过 `--observation-key` 指定的列）
- `frame_index`（可选；如果缺失，会按文件内顺序回退）

### 数据集路径解析

`dataset` 参数支持三种写法：

- 数据集绝对路径
- 相对当前工作目录的路径
- 相对 `main/data/` 的数据集 id，比如 `robocasa/composite/ArrangeBreadBasket`

也可以不用位置参数，改传 `--dataset ...`。

### 默认输出行为

如果不传 `--output-dir`，脚本会直接修改输入数据集：

- 在数据集目录下创建 signature cache
- 更新 `meta/info.json`
- 更新 `meta/stats.json`
- 从 parquet 中移除已存在的 signature 列

如果传了 `--output-dir`，脚本会先把整个数据集复制到目标目录，再在复制后的目录上处理。

### signature cache 布局

默认 cache 目录类似：

```text
<dataset_root>/.signature_cache/<sanitized_dataset_id>/signature_cache_v1/
```

其中通常会生成：

- `observation.path_signature.float16.npy`
- `observation.delta_signature.float16.npy`
- `metadata.json`

如果你改了 `--path-signature-key`、`--delta-signature-key` 或 `--signature-cache-dtype`，文件名也会跟着变化。

### 常见用法

对 `main/data/robocasa/composite/ArrangeBreadBasket` 原地生成 signature cache：

```bash
python data/process_dataset.py robocasa/composite/ArrangeBreadBasket
```

显式使用 `--dataset` 参数：

```bash
python data/process_dataset.py --dataset robocasa/composite/ArrangeBreadBasket
```

输出到新目录，不改原数据：

```bash
python data/process_dataset.py robocasa/composite/ArrangeBreadBasket \
  --output-dir data/robocasa/composite/ArrangeBreadBasket_sig \
  --overwrite-output
```

把 cache 存成 `float32`：

```bash
python data/process_dataset.py robocasa/composite/ArrangeBreadBasket \
  --signature-cache-dtype float32
```

指定别的状态列和 signature key：

```bash
python data/process_dataset.py robocasa/composite/ArrangeBreadBasket \
  --observation-key observation.state \
  --path-signature-key observation.path_signature \
  --delta-signature-key observation.delta_signature
```

把 cache 输出到自定义目录：

```bash
python data/process_dataset.py robocasa/composite/ArrangeBreadBasket \
  --signature-cache-root /tmp/arrange_breadbasket_signature_cache
```

### 常用参数

- `--output-dir`：把处理结果写到新数据集目录；不传时默认原地更新。
- `--overwrite-output`：允许覆盖已存在的 `--output-dir`。
- `--observation-key`：用于计算 signature 的状态向量列，默认是 `observation.state`。
- `--path-signature-key`：写入 `meta/info.json` / `meta/stats.json` 的 path signature 特征名。
- `--delta-signature-key`：写入 `meta/info.json` / `meta/stats.json` 的 delta signature 特征名。
- `--signature-depth`：传给 `signatory.signature` 的截断深度，默认 `3`。
- `--signature-cache-dtype`：cache 存储类型，可选 `float16` 或 `float32`。
- `--signature-cache-root`：自定义 cache 根目录；不传时默认放在目标数据集目录下的隐藏目录里。

### 查看帮助

```bash
python data/process_dataset.py --help
```

## `hfd.sh`

`hfd.sh` 用来下载 Hugging Face 仓库快照，支持模型和数据集两类仓库。

常见场景：

- 下载 LeRobot 数据集到本地
- 指定 `aria2c` 多线程下载
- 通过 `--include` / `--exclude` 过滤需要的文件
- 指定 `--revision` 拉取某个分支、tag 或 commit 对应内容

### 常见用法

下载数据集到当前目录下同名子目录：

```bash
./hfd.sh Yinpei/robomme_data_lerobot --dataset
```

用 `aria2c` 开 8 线程下载：

```bash
./hfd.sh Yinpei/robomme_data_lerobot --dataset --tool aria2c -x 8
```

指定本地输出目录：

```bash
./hfd.sh Yinpei/robomme_data_lerobot --dataset \
  --local-dir robomme_data_lerobot
```

只下载部分文件：

```bash
./hfd.sh Yinpei/robomme_data_lerobot --dataset \
  --include "meta/*" "data/chunk-000/*"
```

排除大文件：

```bash
./hfd.sh Yinpei/robomme_data_lerobot --dataset \
  --exclude "*.mp4"
```

下载指定 revision：

```bash
./hfd.sh Yinpei/robomme_data_lerobot --dataset \
  --revision main
```

### 常用参数

- `--dataset`：把 repo 当作 Hugging Face dataset 下载。
- `--tool aria2c|wget`：选择下载工具，默认 `aria2c`。
- `-x`：`aria2c` 下载线程数，默认 `4`。
- `-j`：`aria2c` 并发下载任务数，默认 `5`。
- `--local-dir`：本地保存目录。
- `--include`：只下载匹配模式的文件。
- `--exclude`：跳过匹配模式的文件。
- `--revision`：指定 revision，默认 `main`。
- `--hf_username` / `--hf_token`：下载 gated 仓库时使用。

### 查看帮助

```bash
bash data/hfd.sh --help
```

## `upload_dataset_to_hf.py`

`upload_dataset_to_hf.py` 用来把 `main/data/` 下的本地数据集目录上传到 Hugging Face 的 `dataset` 仓库。

它会：

1. 解析本地数据集路径，支持 `zeno-ai/xxx`、`data/xxx`、`main/data/xxx` 和绝对路径
2. 自动创建 Hugging Face dataset repo（已存在时复用）
3. 默认使用 `upload_folder()` 单次提交上传
4. 对超大目录可切换到 `upload_large_folder()`，支持断点续传式重试
5. 在真正上传前打印一份解析后的上传计划，方便确认目标路径和 repo id

### 常见用法

上传 `main/data/zeno-ai/CleanTableTopDelayedToolChoice` 到同名 dataset repo：

```bash
python data/upload_dataset_to_hf.py zeno-ai/CleanTableTopDelayedToolChoice
```

显式指定目标 repo：

```bash
python data/upload_dataset_to_hf.py \
  zeno-ai/CleanTableTopDelayedToolChoice \
  --repo-id zeno-ai/CleanTableTopDelayedToolChoice
```

先只查看解析结果，不真正上传：

```bash
python data/upload_dataset_to_hf.py \
  zeno-ai/CleanTableTopDelayedToolChoice \
  --dry-run
```

只上传部分文件：

```bash
python data/upload_dataset_to_hf.py \
  zeno-ai/CleanTableTopDelayedToolChoice \
  --include "meta/*" "data/**/*" \
  --exclude "*.mp4"
```

上传到远端仓库的子目录：

```bash
python data/upload_dataset_to_hf.py \
  zeno-ai/CleanTableTopDelayedToolChoice \
  --repo-id zeno-ai/clean-tabletop-staging \
  --path-in-repo runs/2026-04-17
```

对超大目录使用可恢复的大目录上传：

```bash
python data/upload_dataset_to_hf.py \
  zeno-ai/CleanTableTopDelayedToolChoice \
  --repo-id zeno-ai/clean-tabletop-staging \
  --large-folder \
  --num-workers 16
```

### repo id 推断规则

- 如果本地路径形如 `main/data/<name>`，默认 repo id 会取 `<name>`。
- 如果本地路径形如 `main/data/<namespace>/<name>`，默认 repo id 会取 `<namespace>/<name>`。
- 如果本地路径比这更深，例如 `main/data/robocasa/composite/ArrangeBreadBasket`，请显式传 `--repo-id`，避免远端命名和本地层级不一致。

### 常用参数

- `--repo-id`：显式指定 Hugging Face dataset repo id。
- `--private` / `--public`：设置新建仓库的可见性；仓库已存在时不会改动原设置。
- `--revision`：上传到指定分支、tag 或 revision，默认 `main`。
- `--path-in-repo`：把本地目录上传到远端仓库的某个子目录。
- `--include`：只上传匹配这些 glob 的文件。
- `--exclude`：跳过匹配这些 glob 的文件。
- `--token`：显式传 Hugging Face token；不传时走本地登录态或 `HF_TOKEN`。
- `--large-folder`：改用 `upload_large_folder()`，适合超大目录。
- `--num-workers`：大目录上传时的 worker 数。
- `--dry-run`：只打印解析后的上传计划，不真正创建 repo / 上传。

### 注意事项

- `--large-folder` 不支持 `--path-in-repo`、`--commit-message`、`--commit-description`，因为底层 API 本身就有限制。
- 大目录上传会拆成多个 commit；普通上传则默认是单次 commit。
- 脚本默认 repo 类型固定为 Hugging Face `dataset`。

### 查看帮助

```bash
python data/upload_dataset_to_hf.py --help
```
