# 气象图表多标签分类训练指南

本指南介绍如何使用增强的训练系统进行气象图表多标签分类模型训练。

## 功能特性

### 增强的DataLoader功能
- **状态追踪**: 自动记录训练进度、batch时间、epoch指标
- **检查点保存**: 定期保存训练状态，支持训练恢复
- **实时日志**: 详细的训练进度和性能指标记录
- **实验管理**: 标准化的实验目录结构

### 实验管理系统
- **自动目录创建**: 创建标准化的实验目录结构
- **配置管理**: 支持YAML/JSON配置文件
- **状态跟踪**: 记录实验状态和元数据
- **结果归档**: 自动保存训练结果和指标

## 快速开始

### 1. 基础训练

创建新实验并开始训练：

```bash
# 使用默认配置创建新实验
python train.py --experiment_name "baseline_v1" --description "基础模型训练"

# 使用自定义配置文件
python train.py --experiment_name "baseline_v1" --config config_example.yaml

# 添加实验标签
python train.py --experiment_name "baseline_v1" --tags "baseline" "cnn-rnn" "first_run"
```

### 2. 训练恢复

从中断的训练中恢复：

```bash
# 自动找到最新检查点并恢复
python train.py --experiment_dir ./experiments/baseline_v1_20241201_143022 --resume

# 或者指定具体的实验目录
python train.py --experiment_dir /path/to/experiment --resume
```

### 3. 模型评估

仅运行模型评估：

```bash
# 在验证集上评估
python train.py --experiment_dir ./experiments/baseline_v1_20241201_143022 --evaluate_only
```

### 4. 生成预测

仅生成测试集预测：

```bash
# 在测试集上生成预测
python train.py --experiment_dir ./experiments/baseline_v1_20241201_143022 --predict_only
```

### 5. 调试模式

快速调试和开发：

```bash
# 使用少量数据快速测试
python train.py --experiment_name "debug_test" --fast_dev_run --debug

# 启用详细日志
python train.py --experiment_name "debug_test" --debug
```

## 配置文件

### 配置文件结构

训练系统支持YAML和JSON格式的配置文件。参考 `config_example.yaml` 了解所有可用选项。

### 主要配置节

1. **dataset**: 数据集路径和加载配置
2. **model**: 模型架构参数
3. **training**: 训练超参数
4. **learning_rates**: 学习率配置
5. **optimizer**: 优化器参数
6. **loss_weights**: 损失函数权重
7. **validation**: 验证和早停配置

### 命令行覆盖

可以通过命令行参数覆盖配置文件中的设置：

```bash
python train.py --config config.yaml \
  --num_epochs 100 \
  --batch_size 64 \
  --learning_rate 1e-3
```

## 实验目录结构

每个实验会创建以下标准目录结构：

```
experiments/
└── baseline_v1_20241201_143022/
    ├── checkpoints/           # 模型检查点和训练状态
    │   ├── training_state.json
    │   ├── checkpoint_epoch_010/
    │   ├── checkpoint_epoch_020/
    │   └── best_model/
    ├── logs/                  # 训练日志和指标
    │   ├── epoch_001_metrics.json
    │   ├── epoch_002_metrics.json
    │   └── training_summary.json
    ├── eval_results/          # 评估结果和预测
    │   ├── validation_results.json
    │   ├── test_predictions.json
    │   └── final_test_predictions.json
    ├── configs/               # 配置文件和标签映射
    │   ├── config.yaml
    │   ├── config.json
    │   └── label_mapping.json
    ├── plots/                 # 训练图表和可视化（待实现）
    ├── experiment_metadata.json
    └── README.md
```

## 监控训练进度

### 控制台输出

训练过程中会实时显示：

- Epoch进度和剩余时间估计
- Batch级别的损失和学习率
- 验证指标和最佳模型更新
- 早停和检查点保存信息

### 日志文件

详细日志保存在 `logs/` 目录：

- `epoch_XXX_metrics.json`: 每个epoch的详细指标
- `training_summary.json`: 完整训练总结
- `training.log`: 原始日志文件（在项目根目录）

### 状态文件

训练状态保存在 `checkpoints/training_state.json`：

```json
{
  "current_epoch": 25,
  "current_batch": 150,
  "total_batches": 200,
  "best_metrics": {
    "val_f1": 0.8234,
    "val_accuracy": 0.8567
  },
  "epoch_metrics": [...]
}
```

## 高级功能

### 分布式训练

支持多GPU分布式训练：

```bash
# 单机多卡训练（使用torchrun）
torchrun --nproc_per_node=4 train.py \
  --experiment_name "distributed_baseline" \
  --config config.yaml

# 或使用传统方式
python -m torch.distributed.launch --nproc_per_node=4 train.py \
  --experiment_name "distributed_baseline" \
  --config config.yaml
```

### 实验管理命令

查看和管理实验：

```python
from multi_label_classifier.experiments.experiment_manager import ExperimentManager

# 创建实验管理器
exp_manager = ExperimentManager()

# 列出所有实验
experiments = exp_manager.list_experiments()
for exp in experiments:
    print(f"{exp['full_name']}: {exp['status']}")

# 获取实验详细信息
summary = exp_manager.get_experiment_summary("./experiments/baseline_v1_20241201_143022")
print(summary)

# 归档实验
exp_manager.archive_experiment("./experiments/old_experiment_20241120_100000")
```

### 自定义配置

创建专门的配置文件用于不同实验：

```yaml
# configs/large_model.yaml
model:
  rnn_hidden_dim: 256
  joint_embedding_dim: 512
  rnn_num_layers: 2

training:
  num_epochs: 100
  warmup_epochs: 10

learning_rates:
  rnn_learning_rate: 1.0e-4
  cnn_learning_rate: 1.0e-5
```

```bash
python train.py --experiment_name "large_model_v1" --config configs/large_model.yaml
```

## 故障排除

### 常见问题

1. **内存不足**: 减少batch_size或num_workers
2. **CUDA错误**: 检查GPU可用性和驱动版本
3. **数据加载失败**: 验证数据路径和文件格式
4. **配置错误**: 检查YAML格式和参数类型

### 调试技巧

```bash
# 启用详细调试信息
python train.py --experiment_name "debug" --debug --fast_dev_run

# 检查数据加载
python -c "
from multi_label_classifier.core import DatasetFactory
factory = DatasetFactory('./dataset/multi_label_classifier')
train_dataset = factory.load_dataset('train')
print(f'Dataset size: {len(train_dataset)}')
print(f'Sample: {train_dataset[0]}')
"
```

### 性能优化

1. **数据加载优化**: 调整num_workers和pin_memory
2. **GPU利用率**: 使用更大的batch_size
3. **混合精度**: 启用AMP（在trainer中配置）
4. **检查点频率**: 平衡保存频率和性能

## 最佳实践

1. **实验命名**: 使用描述性名称和版本号
2. **配置管理**: 为不同实验类型创建专门的配置文件
3. **结果记录**: 在README.md中记录重要发现
4. **定期清理**: 归档或删除不需要的实验
5. **资源监控**: 关注GPU内存和磁盘使用情况

## 下一步

- 实现训练可视化和图表生成
- 添加超参数搜索功能
- 集成Weights & Biases或TensorBoard
- 添加模型导出和部署功能
