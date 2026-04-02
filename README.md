# 使用指令

* 数据下载

```bash
python data/download_dataset.py lerobot/metaworld_mt50 --output-dir data
```

* 数据采集

```bash
python scripts/collect_imitation_dataset.py --env braidedhub --num-per-task 100 --path-signature-depth 3 --enable-randomize --enable-first-frame-anchor
```

* 训练

```bash
./bash/train_policy.sh --env braidedhub --policy streaming_act
```