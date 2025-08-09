# extractor模块说明(AI辅助生成)
## 概述

该目录提供对本地 PPTX 文件的图片提取与图像来源分类能力，包含两个核心组件：
- Extractor：从 PPTX 幻灯片中批量提取图片，按规则命名并保存到输出目录。
- SourceClassifier：基于文件名模式与图片内容哈希，判断图像来源（CMA/AOC/NMC/unknown），并导出分类结果。

项目根目录中的 `extract.py` 提供端到端流程：扫描 `slides/` 中的 PPTX → 提取图片到 `extracted_images/` → 分类来源 → 导出 `classified_list.json`。


## 目录结构

```
extractor/
├── __init__.py            # 对外导出入口（Extractor、SourceClassifier）
├── extractor.py           # PPT 图片提取（逐文件提取、批量扫描）
└── source_classifier.py   # 图片来源分类（文件名正则 + 内容哈希）
```

项目根目录中还包含：
- `extract.py`：端到端示例脚本（提取 → 分类 → 导出 JSON）
- `slides/`：输入 PPTX 文件目录（运行后扫描）
- `extracted_images/`：提取图片输出目录（运行后生成）


## 快速开始（推荐）

直接运行项目根目录中的示例脚本：

```bash
python extract.py
```

脚本会：
- 扫描 `slides/` 下的 PPTX 文件
- 将图片提取至 `extracted_images/`
- 对每张图片进行来源分类（cma/aoc/nmc/unknown）
- 基于内容哈希对 NMC 进行二次判定并合并
- 输出分类结果至 `classified_list.json`


## 以代码方式使用（Extractor / SourceClassifier）

- 提取图片：

```python
from extractor import Extractor

extractor = Extractor(output_dir="extracted_images")
ppt_files = extractor.find_pptx_files(input_folder="slides")
for ppt in ppt_files:
    extractor.extract_images_from_ppt(ppt)
```

- 分类来源并导出结果：

```python
import os
from extractor import SourceClassifier

image_dir = "extracted_images"
classifier = SourceClassifier(image_dir=image_dir)

nmc_candidates = []
for name in os.listdir(image_dir):
    path = os.path.join(image_dir, name)
    source = classifier.classify_source(path)  # 返回 "cma" / "aoc" / "nmc" / "unknown"
    if source == "nmc":
        nmc_candidates.append(path)

classifier.check_nmc_image(nmc_candidates)
classifier.save_classified_list("classified_list.json")
```


## 关键行为与可配置点

- 输出与命名
  - 提取输出目录由 `Extractor(output_dir=...)` 指定，默认示例使用 `extracted_images/`。
  - 图片命名规则：`{ppt_name}_slide{slide_index}_img{shape_index}.{ext}`。
    - 例如：`2025年06月03日早间会商首席发言_slide5_img2.png`。

- 来源分类规则（`SourceClassifier`）
  - 基于文件名前缀 `ppt_name` 与正则匹配：
    - CMA（国家气象信息中心）：`^(\d{8})-早会商-信息中心-实况$`（示例：`20250101-早会商-信息中心-实况`）
    - AOC（气象探测中心）：`^aoc(\d{8})$`（示例：`aoc20250101`）
    - NMC（中央气象台）：`^(\d{4})年(\d{2})月(\d{2})日早间会商首席发言$`（示例：`2025年06月03日早间会商首席发言`）
  - `classify_source(image_path)` 在文件名层面返回来源；若不匹配上述正则，返回 `unknown`。

- 内容哈希与二次判定
  - 对图片计算 `sha256`（`compute_hashes`）。
  - `check_nmc_image(image_list)` 会将传入列表中的图片按内容哈希与已有列表对比，若在 AOC/CMA 中存在相同哈希，则将其并入对应集合。

- 数据结构与导出
  - 内部以 `ImageInfo`（Pydantic 模型，字段：`file_path`、`ppt_name`、`slide_index`、`img_index`、`format`）作为键存储哈希；模型开启 `frozen=True` 以保证可哈希性。
  - `save_classified_list(outfile)` 导出 JSON 数组，每个元素包含：
    - `file_path`、`ppt_name`、`slide_index`、`img_index`、`format`、`hash`、`type`（`cma`/`aoc`）。

- 可配置点
  - 输入目录（PPTX）：默认示例为 `slides/`，可自行传参至 `find_pptx_files`。
  - 输出目录（图片）：由 `Extractor(output_dir=...)` 指定。
  - 分类仅依赖文件名前缀的 `ppt_name` 与上述正则；若命名不符合规则将被标记为 `unknown`。


## 常见问题与排查

- 未找到 PPTX / 目录不存在
  - `find_pptx_files()` 若目录不存在或没有 `.pptx` 文件，将记录日志并返回空列表。

- 无法打开 PPTX / 包损坏
  - `Extractor.extract_images_from_ppt()` 捕获 `PackageNotFoundError` 并记录错误日志，请检查 PPTX 文件有效性。

- 分类结果为 `unknown`
  - 多为文件名不满足正则命名规则，检查 PPTX 文件名中用于构造 `ppt_name` 的前缀是否符合规范。

- JSON 导出问题
  - `save_classified_list()` 输出为数组元素对象，不是字典；若需按来源分组，可在加载后自行聚合。


## 日志与输出

- 使用 `logging` 记录关键步骤（扫描、提取、分类、导出）。
- 图片输出位于 `extracted_images/`。
- 分类结果位于项目根目录 `classified_list.json`，示例结构：

```json
[
  {
    "file_path": "/abs/path/extracted_images/2025年06月03日早间会商首席发言_slide5_img2.png",
    "ppt_name": "2025年06月03日早间会商首席发言",
    "slide_index": 5,
    "img_index": 2,
    "format": "png",
    "hash": "<sha256>",
    "type": "cma"
  }
]
```


## 许可

本项目遵循 `LICENSE` 中的开源许可条款。
