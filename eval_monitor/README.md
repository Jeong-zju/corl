# CORL Eval Monitor

零依赖浏览器监控器，用来实时查看 `main/outputs/eval` 下的实验结果。

## 功能

- 自动扫描 `eval/<env+algo>/<run_timestamp>` 目录。
- 区分完整 run、未完成 run、损坏 summary。
- 按环境变体和算法聚合，支持同一组合多轮 run 的比较。
- 用成功率、最佳成功率、平均成功率、加权成功率做对比。
- 查看 per-task 指标与 rollout 视频。
- 浏览器内删除单个 run 或整个实验序列。
- 通过 SSE + 轮询持续刷新。

## 启动

```bash
/home/jeong/zeno/corl/main/bash/run_eval_monitor.sh
```

默认监听 `http://127.0.0.1:8765`。

也可以自定义：

```bash
/home/jeong/zeno/corl/main/bash/run_eval_monitor.sh --host 0.0.0.0 --port 9000
```

## 说明

- 默认监控目录：`/home/jeong/zeno/corl/main/outputs/eval`
- 不依赖 Flask / FastAPI / Node，仅使用 Python 3 标准库。
- 视频通过内置静态服务直接暴露，支持浏览器的 Range 请求。
