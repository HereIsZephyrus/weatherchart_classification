# inspector模块说明(AI辅助生成)
## 概述

该目录提供对数据集的检查与统计能力，包含两个核心组件：
- GalleryInspector：对 ECMWF OpenCharts 下载的图廊数据集进行检查（覆盖率、缺失项、投影分布）。
- PPTInspector：对 PPT 原始文件与提取图片进行检查（来源分布、命名规范、日期范围）。

项目根目录中的 `statistics.py` 提供统一入口：
- `python statistics.py gallery`：检查 `train_data/gallery` 下的图廊数据集，并输出 HTML 报告
- `python statistics.py ppt`：检查 `income` 下的 PPT 与提取图片，并输出 HTML 报告
- `python statistics.py`：执行全部检查，并输出 HTML 报告

## 目录结构

```
inspector/
├── __init__.py               # 对外导出入口（GalleryInspector、PPTInspector）
├── gallery_inspector.py      # 图廊数据集检查（覆盖率、缺失项、投影分布，支持 HTML 报告）
└── ppt_inspector.py          # PPT 与提取图片检查（来源、命名、日期范围，支持 HTML 报告）
```

项目根目录中还包含：
- `statistics.py`：统一入口脚本（支持 gallery/ppt 两种模式，默认生成 HTML 报告）
- `train_data/gallery/`：图廊数据集目录
- `income/`：PPT 相关目录（含 `slides/` 与 `extracted_images/`）
- `reports/`：HTML 报告输出目录（运行后生成）

## GalleryInspector

- 功能：检查图廊数据集的覆盖率、缺失项与投影分布。

- 使用方式：

```python
from inspector import GalleryInspector

inspector = GalleryInspector(base_dir="train_data/gallery")
stats = inspector.inspect()     # 计算统计
inspector.gallery_info()        # 打印简要汇总到日志

# 保存 HTML 报告
html = inspector.to_html()
with open("reports/gallery_report.html", "w", encoding="utf-8") as f:
    f.write(html)
# 或者：
inspector.save_html("reports/gallery_report.html")
```

- 使用方式（脚本）：

```bash
# 仅检查 gallery，并输出 reports/gallery_report.html
python statistics.py gallery
```

- 文件名兼容性说明（已修复路径/命名兼容问题）：
  - 支持以下 `.webp` 文件名格式：
    - `<YYYYMMDDHHMM>_<projection>.webp`（推荐）
    - `<prefix>_<YYYYMMDDHHMM>_<projection>.webp`（历史产物）
    - `<prefix>_<YYYYMMDDHHMM>.webp`（缺少投影时以 `unknown` 记）

- 统计维度：
  - 总量、每类图表计数
  - 每个投影的计数（全局与分类）
  - 日期范围（全局与分类）
  - 缺失投影的日期（按类别）

- 可配置点：
  - 基础目录：`base_dir` 指定数据集位置
  - 期望投影：`projections` 列表（默认东亚、欧亚等四个区域）

## PPTInspector

- 功能：检查 `income/` 下两类数据：
  - `slides/`：PPTX 文件命名、来源分布、日期覆盖范围
  - `extracted_images/`：提取出的图片命名、扩展名分布、来源分布、按 PPT 汇总

- 使用方式（代码）：

```python
from inspector import PPTInspector

ins = PPTInspector(income_dir="income")
slides_stats, images_stats = ins.inspect_all()  # 同时检查 slides 与 images

# 保存 HTML 报告
ins.save_html("reports/ppt_report.html")
# 或获取 HTML 字符串：
html = ins.to_html()
```

- 使用方式（脚本）：

```bash
# 仅检查 PPT，并输出 reports/ppt_report.html
python statistics.py ppt

# 同时检查 gallery 与 PPT
python statistics.py
```

- 命名规则：
  - PPT 文件（不含扩展名的基础名）：
    - CMA：`YYYYMMDD-早会商-信息中心-实况`
    - AOC：`aocYYYYMMDD`
    - NMC：`YYYY年MM月DD日早间会商首席发言`
  - 图片文件：`{ppt_name}_slide{slide_index}_img{img_index}.{ext}`

- 统计维度：
  - Slides：
    - 总量、来源计数（cma/aoc/nmc/unknown）
    - 日期范围（YYYYMMDD）
    - 命名不合规样例
  - Images：
    - 总量、扩展名计数
    - 来源计数（根据 `ppt_name` 推断）
    - 按 PPT 汇总的图片数量（日志展示部分样例）
    - 命名不合规样例

- 可配置点：
  - 输入目录：`income_dir` 指定根目录（默认 `income/`）
  - 来源判定：通过 `PPT_PATTERNS` 正则匹配（可扩展）

## HTML 报告

- 运行脚本后，默认在 `reports/` 目录生成：
  - `gallery_report.html`：图廊检查报告
  - `ppt_report.html`：PPT 与提取图片检查报告

- 报告内容包含：
  - 扫描路径、总量统计
  - 关键维度分布（投影/来源/扩展名等）
  - 日期范围（全局与分类）
  - 缺失项与无效文件名示例（最多展示部分样例）

## 常见问题与排查

- 目录不存在
  - 两个 Inspector 均会记录警告日志并返回空统计
  - 检查目录结构是否符合预期（`train_data/gallery/` 或 `income/{slides,extracted_images}/`）

- 文件名不合规
  - GalleryInspector：检查 `.webp` 文件是否符合上述三种格式之一
  - PPTInspector：检查 PPT 基础名是否匹配来源正则，图片名是否包含完整信息
  - 日志/报告中会展示不合规文件名样例（最多 5 条）

- 统计结果异常
  - 检查日志与报告中的各项统计是否符合预期
  - 对于 PPT，确认来源分布是否与实际情况相符
  - 对于图廊，关注缺失投影的日期列表

## 许可

本项目遵循 `LICENSE` 中的开源许可条款。