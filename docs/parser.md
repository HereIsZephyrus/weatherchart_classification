本工程构建了一个基于 PPT 图像处理的自动化分类系统，整合了文档解析、图像处理与模式识别技术，形成完整的数据处理流水线，各文件功能如下：


project_root/
├── data/                # 输入：待处理PPT文件
├── extracted_images/    # 中间：提取的PPT图像
├── hash_templates/      # 输入：分类模板图像（按类别分目录）
├── outputs/             # 输出：分类结果CSV
│   └── result.csv
├── debug/               # 调试：中间过程数据
│   ├── processed/       # 预处理后的图像
│   └── distances/       # 哈希距离JSON数据
├── utils/
│   ├── ppt_extract.py   # PPT图像提取模块
│   └── hash_utils.py    # 图像处理与分类模块
└── main.py              # 主程序入口


一、各函数功能
1. ppt_extract.py
主要负责从 PPT 文件中提取图片，包含两个核心函数：
extract_images_from_ppt(ppt_path, output_dir)：处理单个 PPT 文件，遍历所有幻灯片中的形状，提取其中的图片并保存到指定目录。图片命名格式为[PPT文件名]_slide[幻灯片序号]_img[图片序号].[格式]，避免重名。
batch_extract_from_folder(input_folder, output_dir)：批量处理文件夹中的所有 PPTX 文件，统计总提取图片数量并输出结果。
此外，文件包含命令行交互逻辑，可通过--input和--output参数指定输入文件夹和输出目录，默认使用项目根目录下的data和extracted_images文件夹。
2. hash_utils.py
提供图片处理、哈希计算和分类识别功能，核心函数包括：
图片预处理：crop_top_left（裁剪左上角区域）、resize_keep_aspect_ratio（保持比例缩放并填充背景）。
哈希计算：compute_hashes和compute_hashes_from_image用于计算图片的 perceptual hash（phash）和 difference hash（dhash）。
模板加载：load_template_hashes从指定目录加载分类模板图片的哈希值，按类别组织。
图片分类：classify_image通过对比待识别图片与模板的哈希距离，确定图片所属类别，返回最佳匹配类别、距离及排序结果。
3. main.py
程序主入口，协调 PPT 图片提取和分类流程：
路径配置：定义输入 PPT 目录、图片输出目录、模板目录等路径参数。
流程控制：加载模板哈希值，遍历指定目录下的 PPT 文件，调用ppt_extract.py提取图片，再调用hash_utils.py对提取的图片进行分类。
结果处理：将分类结果保存到 CSV 文件（outputs/result.csv），同时保存调试信息（预处理图像和距离数据）到对应目录。

二、系统工作流程
1.数据输入阶段
从指定目录（默认data/）递归扫描所有 PPTX 文件，构建待处理文件队列。
2.图像提取阶段
对每个 PPTX 文件执行：
幻灯片级遍历（slide对象迭代）
形状筛选（判断has_image属性）
3.图像二进制流提取与格式解析
按[文件名]_slide[序号]_img[序号]规则存储至extracted_images/
4.特征处理阶段
对提取的图像执行：
预处理：裁剪 ROI（左上角区域）→ 等比缩放至 128×128 → 灰度转换
特征提取：并行计算 pHash 与 dHash 值
模板匹配：与templates/目录下预加载的分类模板哈希库进行比对，取最小汉明距离对应的类别作为识别结果
5.结果输出阶段
结构化结果（文件名、幻灯片号、类别、置信度等）写入outputs/result.csv
中间产物（预处理图像、哈希距离矩阵）保存至调试目录，支持可追溯性分析


