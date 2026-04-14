# 数据目录说明

`main/data/` 里现在主要维护三个工具：

- `convert_legacy_dataset_to_v30.py`：把 legacy 的 LeRobot v2.1 数据集转换成 v3.0 数据集，并带进度条展示转换过程。
- `process_dataset.py`：为已有的 LeRobot 数据集计算全前缀 `path signature` / `delta signature`，把结果写到独立的 signature cache，并同步更新数据集元信息。
- `hfd.sh`：从 Hugging Face 下载 model 或 dataset 仓库快照，适合拉取原始数据集到本地。

## 环境准备

在仓库根目录运行：

```bash
cd /home/jeong/zeno/corl/main
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
