# 气象图表智能分类系统训练技术方案

## 1. 项目概述

### 1.1 核心目标
构建三级联智能分类系统，实现从原始图像到具体气象产品的精准识别：
- **级联1（Gate）**：识别是否为气象图（二分类）
- **级联2（Elements）**：提取气象要素（多标签分类）  
- **级联3（Products）**：识别具体产品类型（单标签分类）

### 1.2 技术路线
- **trainer模块**：CNN-RNN Unified架构，负责要素识别，为产品分类提供辅助信息
- **finetuner模块**：Gemma3-4b多模态微调，基于图像+要素文本实现产品分类
- **增强策略**：水印Logo合成、标题生成，提升模型泛化能力

### 1.3 预期效果
- Gate任务：AUC > 0.95，减少误判
- Elements任务：mAP > 0.80，准确识别气象要素
- Products任务：Top-1 Acc > 0.85，精准产品分类

---

## 2. 数据准备与标注规范

### 2.1 数据源与格式
```bash
train_data/
├── images/           # 统一图像目录
├── labels.jsonl      # 标注文件
└── logos/           # 水印Logo库
```

**标注格式示例**：
```json
{"path": "images/xxx.png", "is_meteo": 1, "labels": ["wind", "pressure"], "product": "24h_cn_2m_max_wind"}
```

### 2.2 标签体系设计
创建 `docs/labels.yaml` 维护统一词汇表：

```yaml
# 要素标签（Elements）
elements:
  - id: wind
    zh: 风场
    en: wind field
    aliases: [风力, 风速]
  - id: pressure  
    zh: 气压场
    en: pressure field
    aliases: [等压线, 气压]
  - id: precipitation
    zh: 降水
    en: precipitation  
    aliases: [降雨, 雷达降水]

# 产品标签（Products）
products:
  - id: 24h_cn_2m_max_wind
    zh: 24小时全国近地面2m最大风速
    elements: [wind]
  - id: 850hpa_pressure_field
    zh: 850hPa气压分布图
    elements: [pressure]
```

### 2.3 数据增强策略
**划分后执行**，避免验证集泄漏：

1. **水印Logo合成**（20-50%样本）
   - 四角随机放置1-2个Logo
   - 透明度0.5-0.8，随机缩放

2. **标题文本生成**
   - 模板：`"24小时全国近地面2m最大风速"`
   - 字体随机采样，字号/颜色轻度变化
   - 放置于图像上方或空白区域

3. **背景噪声模拟**
   - 轻度高斯噪声
   - 压缩伪影模拟

---

## 3. 模型架构与训练策略

### 3.1 级联1：气象图识别（Gate）
**目标**：二分类，过滤非气象图

**方案**：
- 轻量级CNN（ResNet-18/MobileNetV3）
- 输入：224×224，输出：气象图概率
- 阈值：验证集校准，优化F1-Score

### 3.2 级联2：要素识别（CNN-RNN Unified）

#### 模型结构
```
图像 → CNN主干 → 全局特征v → {RNN序列头, BCE并行头} → 要素标签
```

**组件设计**：
- **CNN主干**：ResNet-50（ImageNet预训练），GAP后得到2048维特征
- **标签嵌入**：L个要素类别 + 特殊符号（\<bos\>, \<eos\>），维度256
- **RNN解码器**：双层GRU，hidden_size=384-512，序列预测
- **并行BCE头**：直接回归多热标签，提升召回率

#### 训练目标
```
总损失 = α·L_BCE + β·L_CE + γ·L_coverage
```
- L_BCE：并行多标签二元交叉熵
- L_CE：序列交叉熵（teacher forcing）
- L_coverage：覆盖惩罚，抑制重复预测
- 权重建议：α=1.0, β=0.5, γ=0.1

#### 训练策略
- **预热阶段**：冻结CNN主干5-10轮，仅训练RNN和头部
- **联合训练**：解冻CNN，降低学习率至1e-4
- **标签顺序**：随机打乱同图像内标签，减少顺序偏置
- **Teacher Forcing**：概率从1.0线性衰减至0.5

### 3.3 级联3：产品分类（Gemma3多模态微调）

#### 输入构造
**文本模板**：
```
这是一张气象图。图中包含：{要素列表}。{标题信息}{Logo信息}
请从以下候选中选择最合适的产品名称：{候选产品}
```

**示例**：
```
这是一张气象图。图中包含：风场、气压场。标题包含：24小时、近地面2m、最大风速；带机构水印。
请从以下候选中选择最合适的产品名称：
1. 24小时全国近地面2m最大风速
2. 850hPa气压分布图  
3. 全国降水量分布图
```

#### 模型配置
- **基座**：Gemma3-4b多模态版本
- **视觉塔**：ViT-B/16，分辨率224
- **微调策略**：QLoRA（4-bit量化），rank=8, α=16
- **目标模块**：注意力层Q,K,V,O权重

#### 训练目标
- **受限生成**：候选词表约束，序列交叉熵优化
- **难负采样**：同要素不同时效/高度层作为难例
- **候选筛选**：根据要素预先过滤，降低计算复杂度

---

## 4. 硬件环境优化（3060Ti + 13600KF）

### 4.1 通用配置
```python
# 混合精度训练
torch.backends.cuda.matmul.allow_tf32 = True
scaler = torch.cuda.amp.GradScaler()

# DataLoader优化
dataloader = DataLoader(
    dataset, 
    batch_size=32,
    num_workers=10,         # 13600KF物理核数×0.6-0.8
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
```

### 4.2 任务2优化策略
**内存受限配置**（8GB显存）：
- 图像尺寸：224×224（优先）
- 批大小：32（AMP），不足时降至16
- Gradient Checkpointing：开启
- RNN Hidden：384-512维

### 4.3 任务3优化策略
**大模型微调配置**：
```python
# QLoRA配置
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# LoRA配置  
from peft import LoraConfig
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

**批处理策略**：
- Per-GPU批大小：4
- 梯度累计步数：8（等效批大小32）
- 序列长度：256-512 tokens
- CPU卸载：优化器状态

---

## 5. 训练流程与评估

### 5.1 训练阶段划分

#### 阶段1：数据准备（Week 1-2）
1. 数据清洗与格式统一
2. 构建labels.yaml词汇表
3. 训练/验证/测试集划分（按时间/来源）
4. 水印Logo库准备
5. 零样本基线建立

#### 阶段2：要素识别训练（Week 3-4）
1. CNN-RNN Unified模型训练
2. 超参数调优（学习率、损失权重）
3. 阈值校准（逐类F1最大化）
4. 性能评估与错误分析

#### 阶段3：产品分类微调（Week 5）
1. Gemma3多模态模型微调
2. LoRA权重优化
3. 候选集筛选策略
4. 约束解码实现

#### 阶段4：系统集成（Week 6）
1. 三级联推理流程
2. 端到端性能测试
3. 部署优化与文档
4. A/B测试准备

### 5.2 评估指标体系

| 任务 | 主要指标 | 辅助指标 |
|------|----------|----------|
| Gate | AUC, F1-Score | Precision, Recall, EER |
| Elements | mAP, micro-F1 | macro-F1, per-class AP |
| Products | Top-1/Top-5 Acc | 同要素混淆度, ECE |

### 5.3 模型选择与校准
- **早停策略**：验证集指标连续3轮无提升
- **阈值校准**：网格搜索+F1最大化
- **温度校准**：Platt scaling校正置信度
- **模型融合**：必要时RNN+BCE头加权

---

## 6. 工程实现与部署

### 6.1 代码结构
```
trainer/                    # CNN-RNN训练模块
├── models/                # 模型定义
├── datasets/              # 数据加载
├── losses/                # 损失函数
├── train.py              # 训练脚本
└── evaluate.py           # 评估脚本

finetuner/                 # Gemma3微调模块  
├── multimodal/           # 多模态模型
├── prompts/              # 提示模板
├── finetune.py          # 微调脚本
└── inference.py         # 推理脚本

docs/
├── labels.yaml          # 标签词汇表
└── train.md            # 本技术方案
```

### 6.2 依赖环境
```yaml
dependencies:
  - pytorch>=2.1
  - transformers
  - peft                # LoRA微调
  - bitsandbytes       # 量化训练
  - accelerate         # 分布式训练
  - timm               # 视觉模型
  - wandb              # 实验跟踪
  - scikit-learn       # 评估指标
```

### 6.3 部署优化
- **模型压缩**：4-bit量化推理
- **缓存机制**：视觉特征离线计算
- **批处理**：动态批大小适应
- **错误恢复**：级联失败降级策略

---

## 7. 风险控制与监控

### 7.1 训练风险
- **过拟合**：早停+正则化+数据增强
- **灾难性遗忘**：渐进式解冻+小学习率
- **标签噪声**：置信度阈值+人工复查
- **类别不平衡**：Focal Loss+采样策略

### 7.2 部署监控
- **模型性能**：在线评估指标跟踪
- **数据漂移**：输入分布监控
- **资源使用**：GPU/内存占用告警
- **错误分析**：失败样例收集与分析

---

## 8. 预期成果与扩展

### 8.1 交付物
1. **训练好的模型权重**：Gate、Elements、Products三个模型
2. **阈值配置文件**：thresholds.json
3. **推理代码**：端到端预测API
4. **评估报告**：性能指标与错误分析
5. **部署文档**：环境配置与使用说明

### 8.2 扩展方向
- **新要素扩展**：仅需更新labels.yaml，增量训练
- **新产品增加**：候选词表扩充，LoRA微调
- **多语言支持**：提示模板国际化
- **实时推理**：模型压缩与加速优化

---

## 参考文献

1. Wang et al. "CNN-RNN: A Unified Framework for Multi-label Image Classification" CVPR 2016
2. "Open-Vocabulary Multi-label Image Classification with Pretrained Vision-Language Model"  
3. "Hybrid CNN-RNN + CLIP-Based Pipeline for Enhanced Multi-Label Classification"

---

*本技术方案基于单卡RTX 3060Ti + Intel 13600KF环境设计，可根据实际硬件资源调整配置参数。*
