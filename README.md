# 气象图表要素分类系统

## 系统架构

```
# 核心功能模块
├── crawler/ # 数据爬虫模块, 从ECMWF图廊中爬取气象图表
├── extractor/ # 图片提取模块, 从会商PPT中提取图片, 并进行预处理
├── multi_label_classifier/ # 要素分类模块, 包含数据预处理, CNN-RNN模型和训练流程
├── inspector/ # 数据集检查模块, 对原始数据进行检查并生成报告
├── instructor/ # 图文指令模块, 依托微调的gemma3模型实现图文检索和分类评估
├── craw.py # 数据爬虫模块入口程序
├── extract.py # 图片提取模块入口程序
├── classify.py # 要素分类模块入口程序
└── statistics.py # 数据检查模块入口程序
# 模型训练模块
├── finetuner/ # gemma3:4b模型微调模块, 使用CLIP方法在图表识别和分类任务上进行微调
# 数据流文件
├── income/ # 放置待分类PPT和图片
└── train_data/ # 放置训练需要的数据
        ├── gallery/ # ECMWF图表数据
        ├── radar-dataset/ # 雷达降水数据. 来源: https://huggingface.co/datasets/deepguess/weather-analysis-data
        ├── radar/ # 雷达降水数据, 从radar-dataset中解析的数据库
        └── logo/ # 放置各单位水印
# 文档和配置
├── docker/ # 提供各类封装的Docker配置
├── docs/ # 文档, 包括各模块的说明文档和算法说明
├── README.md # 本文件, 系统说明
├── .gitignore # 忽略文件列表
└── environment.yml # Conda依赖
```

## 系统功能

系统