# Local Eval Comparison Viewer

一个最小可运行的本地 eval 结果对比工具，技术栈只有：

- FastAPI
- Jinja2
- 原生 HTML / JavaScript / CSS

它会扫描 `outputs/eval` 下的 run，并在一个页面里做并排视频对比。

## 功能

- 扫描 `outputs/eval/{env}/{algorithm}/{timestamp}`
- 识别 `.mp4`、`.webm`、`.mov`、`.gif`
- 首页按 `env`、`algorithm`、`timestamp` 过滤
- 每个 run 一列，并排展示
- 每个 run 内按 task 分 tab 展示
- 每个 task 内按 episode 编号排序
- 支持成功 / 失败筛选和分页浏览
- 每个视频卡片直接显示成功或失败
- 每列固定高度，列内滚动，页面整体不无限变长
- 点击卡片中的视频后，弹出原生 `dialog` 放大播放
- 支持重新扫描
- 对缺失目录、空目录、无视频目录有空状态提示

## 运行

在项目根目录执行：

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

默认访问：

```text
http://127.0.0.1:8000
```

## 扫描目录

应用默认优先扫描：

```text
outputs/eval
```

如果这个目录不存在，当前仓库会自动回退到：

```text
main/outputs/eval
```

页面顶部会显示当前实际扫描的根目录。

## 目录约定

期望目录结构：

```text
outputs/
  eval/
    {env_or_dataset_name}/
      {algorithm}/
        {time_stamp}/
          ... files ...
```

## 路由

- `GET /`：首页，支持 query params 过滤
- `POST /rescan`：重新扫描目录并回到首页
- `GET /video/{file_path:path}`：安全返回视频文件

## 备注

- 不使用数据库
- 不使用前端框架
- 不使用 WebSocket
- 不做前后端分离
- 视频列表由服务端模板直接渲染
