# multi_label_classifier模块说明

## 概述

`multi_label_classifier`模块是气象图表智能分类系统的核心，负责实现从图像特征提取到多标签分类的完整流程。它采用CNN-RNN统一框架，能够有效建模气象要素之间的依赖关系，并提供了一整套用于数据预处理、增强和模型训练的工具。

该模块的主要功能包括：
- **数据预处理与增强**：提供`Chart`和`ChartEnhancer`对图表进行抽象和增强。
- **模型核心**：实现`CNN-RNN`统一框架，包含`CNNEncoder`、`RNNDecoder`和`ParallelPredictionHead`。
- **训练与评估**：通过`Trainer`类管理模型的训练、评估和推理流程。
- **配置管理**：使用`ModelConfig`等配置类来定义和管理模型与训练参数。

## 目录结构

```
multi_label_classifier/
├── core/                   # 模型核心组件
│   ├── __init__.py
│   ├── config.py           # 模型与训练配置
│   ├── dataset.py          # 数据集定义
│   ├── model.py            # CNN-RNN模型实现
│   ├── trainer.py          # 训练器
│   └── vocab.py            # 词汇表管理
├── preprocess/             # 数据预处理与生成
│   ├── __init__.py
│   ├── chart.py            # 图表抽象类
│   ├── chart_enhancer.py   # 图表增强器
│   └── dataset_generater.py # 数据集生成器
└── settings.py             # 模块级配置
```

## 核心组件

### `preprocess.Chart`（图表抽象类）

- **功能**：提供图表图像的抽象表示，封装了图像加载、元数据管理等功能。
- **核心特性**：
  - 从文件路径或元数据构建图表对象。
  - 管理图表的元数据，如产品名称、标签等。
  - 支持图像的保存和格式转换。

### `preprocess.ChartEnhancer`（图表增强器）

- **功能**：对`Chart`对象进行多种图像增强，以提升模型的泛化能力。
- **核心增强技术**：
  - **尺寸调整**：裁剪和缩放到目标尺寸。
  - **颜色调整**：色调、对比度、亮度、饱和度调节。
  - **水印添加**：随机添加机构Logo。
  - **标题叠加**：在图像顶部添加标题文字。
- **预设配置**：提供多种预设的增强配置（如`BalancedEnhance`、`WeatherAppStyle`等）。

### `core.model.CNNEncoder`（CNN编码器）

- **功能**：使用ResNet-50从输入图像中提取高级视觉特征。
- **核心特性**：
  - 基于ImageNet预训练的`ResNet-50`骨干网络。
  - 通过全局平均池化（GAP）生成2048维的特征向量。
  - 将特征投影到256维的联合嵌入空间。
  - 支持8位量化，以减少模型体积和加速推理。

### `core.model.RNNDecoder`（RNN解码器）

- **功能**：基于`method_CNN-RNN.md`中的理论，将多标签分类任务建模为序列预测问题。
- **核心特性**：
  - 自定义的LSTM实现，用于捕获标签间的依赖关系。
  - 在联合嵌入空间中融合图像特征和上一时刻的标签预测。
  - 逐个生成序列中的标签。

### `core.model.ParallelPredictionHead`（并行预测头）

- **功能**：直接从图像特征预测一个多热编码的标签向量。
- **核心特性**：
  - 作为序列预测的补充，提供全局的、不考虑顺序的标签预测。
  - 改善模型的召回率，并在训练初期提供稳定的监督信号。

### `core.model.WeatherChartModel`（统一模型）

- **功能**：整合`CNNEncoder`、`RNNDecoder`和`ParallelPredictionHead`，形成完整的CNN-RNN统一框架。
- **核心特性**：
  - **双重预测机制**：同时输出序列预测和并行预测的结果。
  - **训练模式**：支持`teacher forcing`机制。
  - **推理模式**：采用`beam search`算法生成最优的标签序列。
  - **量化支持**：提供了完整的8位量化流程，包括准备、校准和转换。

### `core.trainer.Trainer`

- **功能**：管理模型的整个生命周期，包括训练、评估和推理。
- **核心特性**：
  - **分阶段训练**：支持先预热（冻结CNN）再端到端微调的训练策略。
  - **混合损失**：结合序列交叉熵损失和并行BCE损失进行优化。
  - **学习率调度**：集成`CosineAnnealingLR`等学习率调度器。
  - **日志与监控**：通过`TensorBoard`记录训练过程中的各项指标。

## 使用方式

### 模型训练

通过`Trainer`类可以方便地启动一个训练任务。

```python
from multi_label_classifier.core.config import ModelConfig, TrainingConfig
from multi_label_classifier.core.dataset import WeatherDataset
from multi_label_classifier.core.model import WeatherChartModel
from multi_label_classifier.core.trainer import Trainer

# 1. 加载配置
model_config = ModelConfig.from_pretrained("path/to/config.json")
training_config = TrainingConfig(output_dir="experiments/my_experiment")

# 2. 准备数据集
train_dataset = WeatherDataset(data_path="path/to/train.jsonl", config=model_config)
eval_dataset = WeatherDataset(data_path="path/to/eval.jsonl", config=model_config)

# 3. 初始化模型
model = WeatherChartModel(config=model_config)

# 4. 初始化训练器
trainer = Trainer(
    model=model,
    args=training_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 5. 开始训练
trainer.train()
```

### 模型推理

训练完成后，可以使用`Trainer`或直接加载模型进行推理。

```python
from PIL import Image
from torchvision import transforms
from multi_label_classifier.core.model import WeatherChartModel
from multi_label_classifier.core.vocab import vocabulary

# 加载模型
model = WeatherChartModel.from_pretrained("path/to/best_model")
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor(),
])
image = Image.open("path/to/image.png").convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# 模型推理
outputs = model.generate(images=image_tensor)
sequence = outputs["sequences"][0].tolist()

# 解码结果
labels = vocabulary.decode(sequence)
print(f"Predicted labels: {labels}")
```

## 配置说明

模块的行为由`core/config.py`中的配置类控制。

- `ModelConfig`: 定义了模型的超参数。
  - `cnn_config`: CNN编码器的配置，如骨干网络、输出维度等。
  - `rnn_config`: RNN解码器的配置，如隐藏层维度、dropout率等。
  - `quantization_config`: 量化相关的配置。
- `TrainingConfig`: 继承自`transformers.TrainingArguments`，定义了训练过程的参数，如学习率、批大小、训练轮数等。

可以方便地通过`json`文件加载和保存这些配置，以实现实验的可复现性。
