# inspector模块说明(AI辅助生成)
## 概述

inspector模块提供了完整的数据集检查、处理与生成能力，包含六个核心组件：

### 数据集检查与统计
- **GalleryInspector**：对 ECMWF OpenCharts 下载的图廊数据集进行检查（覆盖率、缺失项、投影分布）
- **PPTInspector**：对 PPT 原始文件与提取图片进行检查（来源分布、命名规范、日期范围）

### 数据处理与增强
- **Chart**：图表抽象类，提供图像加载、元数据管理和标题生成功能
- **ChartEnhancer**：图表增强器，支持多种图像增强技术（尺寸调整、颜色调整、水印添加等）
- **RadarDatasetParser**：雷达数据集解析器，用于处理Hugging Face上的天气分析数据集

### 数据集生成与管理
- **DatasetManager & DataBatchBuilder**：完整的数据集生成管理系统，支持批量数据生成、训练/验证/测试集划分

项目根目录中的 `statistics.py` 提供统一入口：
- `python statistics.py gallery`：检查 `train_data/gallery` 下的图廊数据集，并输出 HTML 报告
- `python statistics.py ppt`：检查 `income` 下的 PPT 与提取图片，并输出 HTML 报告
- `python statistics.py`：执行全部检查，并输出 HTML 报告

## 目录结构

```
inspector/
├── __init__.py               # 对外导出入口（所有主要类）
├── gallery_inspector.py      # 图廊数据集检查（覆盖率、缺失项、投影分布，支持 HTML 报告）
├── ppt_inspector.py          # PPT 与提取图片检查（来源、命名、日期范围，支持 HTML 报告）
├── chart.py                  # 图表抽象类（图像加载、元数据管理、标题生成）
├── chart_enhancer.py         # 图表增强器（颜色调整、水印、标题、剪裁等增强技术）
├── dataset_maker.py          # 数据集生成管理（批量生成、配置管理、数据集划分）
└── radardata_parser.py       # 雷达数据集解析器（Hugging Face 数据集处理）
```

项目根目录中还包含：
- `statistics.py`：统一入口脚本（支持 gallery/ppt 两种模式，默认生成 HTML 报告）
- `data_maker.py`：数据集生成管理脚本（支持批量生成、配置管理、数据集划分）
- `parse_radar_data.py`：雷达数据集解析器脚本（支持Hugging Face数据集处理）
- `train_data/`：初始数据集目录
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

## Chart（图表抽象类）

- **功能**：提供图表图像的抽象表示，包含图像加载、元数据管理和标题生成等核心功能。

- **核心特性**：
  - 自动从文件路径解析图表元数据（产品名称、时间信息、分类标签）
  - 支持从双语映射文件加载中英文产品名称
  - 提供随机标题生成功能（用于数据增强）
  - 支持从数据集信息直接构建元数据

- **使用方式**：

```python
from inspector import Chart

# 从图像文件创建图表实例
chart = Chart(image_path="path/to/image.webp", index=0)

# 访问图表元数据
metadata = chart.metadata
print(f"产品名称: {metadata['zh_name']}")
print(f"分类标签: {metadata['label']}")

# 生成随机标题
title = chart.construct_title()
print(f"生成标题: {title}")

# 保存图表（转换为RGB格式）
chart.save("output/chart.png")
```

- **元数据结构**：
  - `index`：图表索引
  - `en_name`：英文产品名称
  - `zh_name`：中文产品名称（可选）
  - `label`：分类标签列表
  - `summary`：产品描述（可选）

## ChartEnhancer（图表增强器）

- **功能**：对图表图像进行多种增强处理，提升模型的泛化能力和鲁棒性。

- **核心增强技术**：
  - **尺寸调整**：智能裁剪和缩放到目标尺寸
  - **图表区域剪裁**：模拟真实使用场景中的局部截图
  - **颜色调整**：色调偏移、对比度、亮度、饱和度调节
  - **水印添加**：随机位置添加机构Logo水印
  - **标题叠加**：在图像顶部添加带背景的标题文字

- **预设配置**：提供11种预设增强配置，适应不同场景需求
  - `None`：仅调整尺寸，不进行其他增强
  - `BalancedEnhance`：平衡的增强配置
  - `HighWatermarkStable`：高水印概率，稳定的图像质量
  - `WeatherAppStyle`：模拟天气应用的显示风格
  - `PresentationReady`：适合演示的高质量配置
  - 其他专用场景配置...

- **使用方式**：

```python
from inspector import ChartEnhancer, EnhancerConfigPresets, Chart

# 使用预设配置
enhancer = ChartEnhancer(EnhancerConfigPresets["BalancedEnhance"])

# 或创建自定义配置
from inspector.chart_enhancer import EnhancerConfig
custom_config = EnhancerConfig(
    use_clip=True,
    add_logo_prob=0.5,
    add_title_prob=0.6,
    clip_chart_prob=0.3,
    hue_shift_prob=0.2,
    contrast_prob=0.3,
    brightness_prob=0.2,
    saturation_prob=0.3
)
enhancer = ChartEnhancer(custom_config)

# 应用增强
chart = Chart("input/image.webp", index=0)
enhanced_chart = enhancer.enhance(chart)
enhanced_chart.save("output/enhanced.png")
```

## DatasetManager & DataBatchBuilder（数据集生成管理）

- **功能**：完整的数据集生成和管理系统，支持大规模批量数据生成、自动训练/验证/测试集划分。

- **核心组件**：
  - **DatasetManager**：数据集管理器，负责整体配置和批次管理
  - **DataBatchBuilder**：批次构建器，负责单个批次的数据生成
  - **DatasetConfig**：配置类，定义数据集生成参数

- **主要特性**：
  - 支持多源数据融合（ECMWF图廊数据 + Hugging Face雷达数据）
  - 自动数据集划分（训练/验证/测试）
  - 随机增强策略应用
  - 批量元数据管理
  - 进度跟踪和状态管理

- **使用方式**：

```python
from inspector import DatasetManager, DatasetConfig

# 配置数据集生成参数
config = DatasetConfig(
    batch_num=50,              # 总批次数
    single_batch_size=1000,    # 每批次样本数
    train_percent=0.7,         # 训练集比例
    validation_percent=0.1,    # 验证集比例
    test_percent=0.2          # 测试集比例
)

# 创建数据集管理器并生成数据集
manager = DatasetManager(config)
manager.build_dataset()  # 执行完整的数据集生成流程
```

- **输出结构**：
```
dataset/
├── train_batch_0000/
│   ├── images/          # 图像文件
│   ├── labels.json      # 标签元数据
│   └── config.json      # 增强配置
├── train_batch_0001/
├── ...
├── validation_batch_XXXX/
└── test_batch_XXXX/
```

## RadarDatasetParser（雷达数据集解析器）

- **功能**：解析和转换Hugging Face上的天气分析数据集，转换为本地可用的图像-标签格式。

- **支持数据集**：
  - `deepguess/weather-analysis-dataset`：天气分析数据集

- **转换功能**：
  - 从Hugging Face自动下载和缓存数据集
  - 提取图像数据并保存为PNG格式
  - 生成对应的标签CSV文件（包含产品类型、上下文摘要、可见参数等）
  - 过滤无效样本（缺少可见参数的数据）

- **使用方式**：

```python
from inspector import RadarDatasetParser

# 创建解析器并执行转换
parser = RadarDatasetParser()
parser.convert_dataset()  # 完整的数据集转换流程
```

- **输出结构**：
```
train_data/radar/
├── images/
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── labels.csv           # 包含索引、产品类型、摘要、特征等信息
```

- **标签字段**：
  - `index`：样本索引
  - `en`：产品类型（英文）
  - `summary`：上下文摘要
  - `feature`：可见参数列表

## 扩展的使用示例

### 完整的数据处理流程

```python
from inspector import (
    RadarDatasetParser, 
    DatasetManager, 
    DatasetConfig,
    Chart,
    ChartEnhancer,
    EnhancerConfigPresets
)

# 1. 准备雷达数据集
parser = RadarDatasetParser()
parser.convert_dataset()

# 2. 配置数据集生成
config = DatasetConfig(
    batch_num=20,
    single_batch_size=500,
    train_percent=0.8,
    validation_percent=0.1,
    test_percent=0.1
)

# 3. 生成训练数据集
manager = DatasetManager(config)
manager.build_dataset()

# 4. 单独处理图表（可选）
chart = Chart("path/to/chart.webp", index=0)
enhancer = ChartEnhancer(EnhancerConfigPresets["WeatherAppStyle"])
enhanced_chart = enhancer(chart)
enhanced_chart.save("enhanced_output.png")
```

### 批量图像增强

```python
import os
from pathlib import Path
from inspector import Chart, ChartEnhancer, EnhancerConfigPresets

# 批量处理文件夹中的图像
input_dir = Path("input_images")
output_dir = Path("enhanced_images")
output_dir.mkdir(exist_ok=True)

enhancer = ChartEnhancer(EnhancerConfigPresets["BalancedEnhance"])

for i, image_path in enumerate(input_dir.glob("*.webp")):
    chart = Chart(str(image_path), index=i)
    enhanced = enhancer(chart)
    enhanced.save(output_dir / f"enhanced_{i:04d}.png")
    print(f"处理完成: {image_path.name}")
```

## 许可

本项目遵循 `LICENSE` 中的开源许可条款。