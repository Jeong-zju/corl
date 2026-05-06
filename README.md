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
python scripts/collect_imitation_dataset.py --env braidedhub --num-per-task 100 --path-signature-depth 3 --enable-randomize
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

* 统一 eval 接口说明

评估入口、公共参数、RoboCasa task 自动解析规则见：

```text
EVAL_INTERFACE.md
```

RoboCasa 现在支持下面这几种统一写法：

```bash
./bash/eval_policy.sh --env robocasa --policy act --dataset robocasa/composite/ArrangeBreadBasket --policy-path <ckpt_dir>
./bash/eval_policy.sh --env robocasa --policy diffusion --dataset robocasa/atomic/CloseFridge --policy-path <ckpt_dir>
./bash/eval_policy.sh --env robocasa --policy streaming_act --dataset robocasa/composite --policy-path <ckpt_dir>
./bash/eval_policy.sh --env robocasa --policy act --tasks ArrangeBreadBasket,PickPlaceCounterToSink --policy-path <ckpt_dir>
./bash/eval_policy.sh --env robocasa --policy act --dataset robocasa/composite/ArrangeBreadBasket --robocasa-split all --policy-path <ckpt_dir>
```

* 开启 eval monitor

```bash
cd monitor
pip install -r requirements.txt
uvicorn app:app --reload
```


# 部署指令

```bash
git clone https://github.com/Jeong-zju/corl.git
cd corl/
git switch develop/benchmark
(python==3.12.13)
pip install -r requirements.txt
pip install -e depends/signatory --no-build-isolation
sudo apt install aria2
hf auth login
wandb login
cd data
./hfd.sh zeno-ai/CleanTableTopDelayedToolChoice --dataset --local-dir zeno-ai/CleanTableTopDelayedToolChoice --hf_username jeong-zju --hf_token <token>
./hfd.sh zeno-ai/BookOriginRelocation --dataset --local-dir zeno-ai/BookOriginRelocation --hf_username jeong-zju --hf_token <token>
cd ..
python data/process_dataset.py zeno-ai/CleanTableTopDelayedToolChoice
python data/process_dataset.py zeno-ai/BookOriginRelocation
bash bash/train_policy.sh --dataset zeno-ai/CleanTableTopDelayedToolChoice --policy streaming_act
bash bash/train_policy.sh --dataset zeno-ai/BookOriginRelocation --policy streaming_act
vim bash/defaults/zeno-ai/CleanTableTopDelayedToolChoice/streaming_act.yaml
vim bash/defaults/zeno-ai/BookOriginRelocation/streaming_act.yaml

(new terminal)

python scripts/upload_checkpoints_to_hf.py --train-output-root outputs/train/zeno-ai/CleanTableTopDelayedToolChoice/streaming-act-prism --repo-id zeno-ai/CleanTableTopDelayedToolChoice-streaming-act --mode full --watch

python scripts/upload_checkpoints_to_hf.py --train-output-root outputs/train/zeno-ai/BookOriginRelocation/streaming-act-prism --repo-id zeno-ai/BookOriginRelocation-streaming-act --mode full --watch
```

# Test

我给你详细描述一下我的最新的任务设置和任务目的，需要你给我这个任务设计一个贴切的名字用于在论文中展示。



【任务目的】

检验PRISM算法的效果。

【任务描述】

整体环境设置在一个酒店房间中，任务开始阶段，相机看见不同位置的书本然后运动机械臂夹取（三个位置只能同时有一个位置有书本）。根据夹取位置的不同，机器人走过类似的路径分别将书本放在房间中三个不同的位置，目标放置位置由起始夹取位置唯一决定。

【任务难点】

不仅仅是长序列任务，而且需要记忆机制来帮助policy做出后续决定。

【任务数据集路径】

/home/jeong/zeno/corl/main/data/zeno-ai/mem-book



git clone https://github.com/Jeong-zju/corl.git && cd corl/ && git switch develop/benchmark

vim bash/defaults/zeno-ai/BookOriginRelocation/streaming_act.yaml

bash bash/install_deploy_zeno.sh --dataset zeno-ai/BookOriginRelocation --hf-token <hf-token> --wandb-token <wandb-token>
