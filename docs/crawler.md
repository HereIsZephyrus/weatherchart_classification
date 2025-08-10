# crawler模块说明(AI辅助生成)
## 概述

该目录提供对 ECMWF OpenCharts（`https://charts.ecmwf.int`）的“图廊（gallery）”抓取与筛选能力，可自动：
- 通过浏览器自动化进入图廊页，按参数勾选筛选项
- 扫描当前页可见卡片并收集具体图表链接
- 将跨多类别重复出现的链接统一重组
- 根据链接批量打开具体图表页面，下载对应的图像（webp）

## 目录结构

```
crawler/
├── __init__.py               # 对外导出入口（Crawler、Driver、GallerySelector、GalleryCrawler、ChartCrawler 等）
├── driver.py                 # 浏览器驱动封装（Selenium Chrome，无头）
├── gallery_selector.py       # 图廊筛选器（勾选 MUI Checkbox 实现筛选）
├── gallery_crawler.py        # 图廊页解析（计算行数、按行抓取卡片 href、滚动）
├── crawler.py                # 统一编排（初始化、应用筛选、提取链接、重组分类）
└── chart_crawler.py          # 具体图表下载（按时间/投影拼接参数并下载 webp）
```

项目根目录中还包含：
- `craw.py`：端到端示例脚本（筛选→收集链接→并发下载）
- `gallery/`：下载输出目录（运行后生成）
- `debug_html/`：调试页面快照（需要时生成）


Linux 服务器环境缺少浏览器时，可按需安装 Chrome/Chromium。


## 快速开始（推荐）

直接运行项目根目录中的示例脚本：

```bash
python craw.py
```

脚本会：
- 依次使用参数列表对图廊执行筛选
- 收集每个参数对应的图表链接
- 调用 `Crawler.reorganize_gallery` 合并跨类重复链接
- 使用进程池并发调用 `download_gallery_task` 下载图像

下载产物默认位于：
```
gallery/<分类名>/<YYYYMMDDHHMM>_<projection>.webp
```


## 以代码方式使用（Crawler / 下载 / 统计）

- 初始化与筛选、收集链接：

```python
from crawler import Crawler

crawler = Crawler()
# 仅以“Wind”为例
crawler.filter(["Wind"])                   # 应用筛选（内部会固定 Range/Type/Component/Product type）
hrefs = crawler.extract_chart_hrefs()       # 返回当前页可见卡片的链接列表
```

- 重组分类（去重并将多类同时出现的链接合并到复合键）：

```python
from typing import Dict, List

gallery: Dict[str, List[str]] = {"Wind": hrefs, "Temperature": [...]}  # 示例
reorg = crawler.reorganize_gallery(gallery)
# reorg 的键可能是单类（如 wind）或复合键（多类以 'A' 连接，如 atmosphereAtemperature）
```

- 按链接下载图表（带时间/投影枚举）：

```python
from crawler import download_gallery_task

# kind 用于输出子目录名
download_gallery_task(kind="wind", urls=reorg["wind"])  # 输出到 gallery/wind/
```

## 关键行为与可配置点

- 浏览器与等待
  - `driver.Driver` 默认无头启动，窗口大小 1920×1080，超时 `wait_timeout=30s`；可在实例化时传入 `wait_timeout`、`user_agent`（当前 `Crawler()` 与 `ChartCrawler()` 里使用默认值）。
  - `Driver.wait_for_page_load()` 先等 `document.readyState === 'complete'`，若存在 jQuery 再等待 AJAX 空闲（最多 5s）。
  - `Driver.save_html(filepath=None)` 可将当前页 HTML 保存到 `debug_html/` 便于排查。

- 图廊筛选
  - `GallerySelector._build_filter_categories()` 固定：
    - Range: Medium (15 days)
    - Type: Forecasts
    - Component: Surface, Atmosphere
    - Product type: Control Forecast (ex-HRES), Ensemble forecast (ENS), Extreme forecast index, AIFS Single, AIFS Ensemble forecast, AIFS ENS Control
    - Parameters: 由调用方传入的参数列表（如 Wind、Temperature 等）
  - 筛选定位依赖页面的 MUI 类名选择器（可能随网站改版变动），若报元素未找到请参见“故障排查”。

- 图廊解析
  - `GalleryCrawler.get_number_of_rows()` 通过容器高度与卡片高度（520px）估算行数。
  - `get_href_line(index)` 按行定位 `a[role='button']` 收集 href。
  - `get_next_row_count(index, current_row_count)` 以 3 行为步长滚动以加载更多。

- 图表下载
  - `ChartCrawler` 为每个基础链接枚举：
    - 时间：从当天 00:00 起往回每 6 小时共 30 个时刻（`YYYYMMDDHHMM`）
    - 投影：`opencharts_eastern_asia`、`opencharts_eruasia`、`opencharts_south_east_asia_and_indonesia`、`opencharts_southern_asia`
  - 构造 URL：`{base_url}?base_time={t}&valid_time={t}&projection={p}`，打开后抓取首个 `img` 的 `src`，再用 `requests` 下载。
  - 输出文件名：`{date}_{projection}.webp`。
  - 如需减少下载量，可修改 `ChartCrawler.date_range` 与 `projection_list`。


## 常见问题与排查

- 元素找不到/筛选失效
  - 站点前端为 MUI，类名可能变动。若 `GallerySelector` 抛出超时/未找到，请更新 `gallery_selector.py` 里的选择器（`legend`、`input` 等）或放宽定位策略。

- 超时/空白页
  - 服务器网络慢或资源紧张时可提高 `Driver(wait_timeout=...)`，或在关键步骤增加 `wait_for_update(timedelay=...)` 的延时。

- 浏览器或驱动不可用
  - 确认 Chrome/Chromium 可运行；在容器/CI 环境需添加 `--no-sandbox`、`--disable-dev-shm-usage`（已在代码里默认启用）。

- 下载失败
  - 日志中若出现“Failed to get image url/data”，多为网络波动或目标页面未加载完全。可重试或适当增大等待时间。


## 日志与输出

- 通过 `logging` 记录关键步骤（初始化、页面加载、行数统计、链接/图片下载等）。
- 下载输出位于 `gallery/<分类>/`；调试页面可保存到 `debug_html/`。


## 许可

本项目遵循 `LICENSE` 中的开源许可条款。
