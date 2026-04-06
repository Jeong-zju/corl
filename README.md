# 使用指令

* 数据下载

```bash
python data/download_dataset.py lerobot/metaworld_mt50 --output-dir data
```

* Meta-World MT50 的 SIPM 数据预处理

```bash
bash bash/prepare_metaworld_mt50_sipm_dataset.sh --overwrite-output
```

如果你本地已经有 `lerobot_metaworld_mt50_video`，也可以显式指定：

```bash
bash bash/prepare_metaworld_mt50_sipm_dataset.sh \
  --source-dataset lerobot_metaworld_mt50_video \
  --target-dataset data/lerobot_metaworld_mt50_video_sipm \
  --overwrite-output
```

* 数据采集

```bash
python scripts/collect_imitation_dataset.py --env braidedhub --num-per-task 100 --path-signature-depth 3 --enable-randomize --enable-first-frame-anchor
```

* 训练

```bash
./bash/train_policy.sh --env braidedhub --policy streaming_act
./bash/train_policy.sh --dataset metaworld_mt50 --policy act
./bash/train_policy.sh --dataset metaworld_mt50 --policy streaming_act

lerobot-train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --dataset.repo_id=local/metaworld_mt50 \
  --dataset.root=$PWD/data/lerobot_metaworld_mt50/ \
  --env.type=metaworld \
  --env.task=assembly-v3,dial-turn-v3,handle-press-side-v3 \
  --output_dir=$PWD/outputs/train/${RUN_NAME} \
  --steps=100000 \
  --batch_size=8 \
  --num_workers=20 \
  --save_freq=5000 \
  --eval_freq=-1
```

`metaworld` 的默认配置位于 `bash/defaults/metaworld/`：

- `act` 默认读取 `data/lerobot_metaworld_mt50_video_sipm`
- `streaming_act` 默认读取 `data/lerobot_metaworld_mt50_video_sipm`

* Meta-World 环境 rollout eval（输出视频和成功率）

```bash
./bash/eval_policy.sh --env metaworld --policy act
./bash/eval_policy.sh --env metaworld --policy streaming_act
./bash/eval_policy.sh --env metaworld --policy act --task assembly-v3,dial-turn-v3,handle-press-side-v3
```

默认的 Meta-World task 子集是 `assembly-v3,dial-turn-v3,handle-press-side-v3`，并且默认每个 task 跑 `50` 条 rollout、导出 `50` 条视频。环境 rollout eval 会在 `outputs/eval/...` 下输出 `videos/`、`eval_info.json` 和 `summary.json`。

* 离线 held-out eval

```bash
./bash/eval_policy.sh --dataset metaworld_mt50 --policy act --policy-path <ckpt_dir>
./bash/eval_policy.sh --dataset metaworld_mt50 --policy streaming_act --policy-path <ckpt_dir>
```

* 开启 eval monitor

```bash
cd monitor
pip install -r requirements.txt
uvicorn app:app --reload
```
