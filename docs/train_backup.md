## 气象图多标签分类训练技术路线（trainer 与 finetuner）

本文档给出项目的端到端技术路线、模型选择与训练方法，覆盖二阶段任务（先判别是否为气象图，再进行要素多标签识别）、两条模型路线（CNN‑RNN Unified 与 CLIP/Gemma3 视觉语言微调）、融合与阈值校准、评估与上线注意事项。

### 1. 任务定义与标签体系（三阶段级联）

- **任务1（Gate）**：二分类，输入一张图片，判断是否为气象图（正类）。输出 `p_meteo ∈ [0,1]`。
- **任务2（Elements, CNN‑RNN Unified）**：若为气象图，则进行多标签要素识别（如云图、降水、气压等），一次可预测多个要素。输出 `ŷ ∈ [0,1]^L`，其中 `L` 为要素类别数。该模块将作为任务3的辅助信号来源。
- **任务3（Products, Gemma3 微调）**：给定图片与由任务2输出的要素标签（文本化）作为条件，识别该气象图属于哪种具体产品（单标签多类，如“24小时全国近地面2m最大风速”“850hPa气压分布图”等）。输出 `ŷ_prod ∈ {1..C}`，其中 `C` 为产品类别数。

标签建议：在 `docs/labels.yaml`（或 `train_data/labels.yaml`）中维护统一标签清单与同义词映射，兼容中英文名称与提示词（prompt）。示例结构：

```yaml
# 示例：请结合真实数据补充完整
binary:
  positive: meteorological_chart
elements:
  - id: cloud_map
    zh: 云图
    en: cloud imagery
    aliases: [卫星云图, 云系]
  - id: precipitation
    zh: 降水
    en: precipitation
    aliases: [雷达降水, 降雨]
  - id: pressure
    zh: 气压场
    en: pressure field
    aliases: [等压线, isobar]
  # ... 其余要素
```

数据标注格式建议使用 JSONL：

```json
{"path": "train_data/gallery/xxx.png", "is_meteo": 1, "labels": ["cloud_map", "pressure"]}
{"path": "train_data/gallery/yyy.png", "is_meteo": 1, "labels": ["wind", "pressure"], "product": "24h_cn_2m_max_wind"}
```

### 2. 数据准备与划分（含水印与标题增强）

- 数据来源：`train_data/gallery/`、`train_data/rader-dataset/` 等目录；按项目需要统一为 `images/` 与 `labels.jsonl`。
- 划分：`train/val/test`（按图表来源与时间切分，避免泄漏）。
- 类别不均衡：
  - 使用 `class_weight` 或 `pos_weight`；
  - 采用 Focal Loss（γ=2, α=0.25 作为起点）；
  - 采样器：类别均衡采样 + 正负比例控制（Gate 任务建议 1:1 至 1:3）。
- 预处理：等比缩放短边至 256～320，中心或随机裁剪至 224/256；保留图例区域信息可增益要素识别（建议不裁掉图例）。
- 增强（划分后执行，以避免验证/测试泄漏）：
  - 水印与 Logo 合成：从 `train_data/logo/` 抽样 0.2～0.5 的训练样本，在四角随机放置 1～2 个 Logo（随机缩放与透明度 0.5～0.8），记录位置信息；
  - 标题/注释合成：在图像上方或空白区域渲染随机生成的中英文标题，标题模板包含要素词与产品词汇（如“24小时全国近地面2m最大风速”“850hPa气压分布图”），字体从候选列表中随机采样，字号/颜色轻度抖动；
  - 背景轻度噪声与压缩伪影模拟，提升鲁棒性。

### 3. 总体方案（三级联 + 双路可选融合）

1) 级联推理：
- 模型A（Gate）：二分类器。若 `p_meteo < τ_gate`（验证集校准），直接输出“非气象图”，跳过后续步骤。
- 模型B（Elements，CNN‑RNN Unified）：对被判定为气象图的样本进行多标签预测，得到要素集合 `S_elem`（通过阈值化 `ŷ` 获得）。
- 模型C（Products，Gemma3 微调）：输入为图像 + 文本描述（由 `S_elem` 文本化而来，可附带合成标题/Logo 信息），输出产品类别。

2) 双路融合（可选）：
- 若产品分类（任务3）仅依赖 Gemma3，可不与 CLIP 路线融合；
- 若需要提升要素识别鲁棒性，可并行引入 CLIP 路线进行要素多标签的辅助打分，并与 CNN‑RNN 在任务2阶段融合（见第7节）。

---

### 4. trainer 模块：CNN‑RNN Unified（多标签）

参考：`docs/literature/Wang_CNN-RNN_A_Unified_CVPR_2016_paper.pdf`

#### 4.1 模型结构

- CNN 主干：ResNet‑50/101（ImageNet 预训练），输出全局图像特征 `v ∈ R^d`（GAP 后 d=2048），或保留 `7×7×2048` 的空间特征做注意力扩展（可选）。
- 标签嵌入：为 `L` 个要素类别 + 特殊符号（`<bos>`, `<eos>`, `<pad>`）学习 `E ∈ R^{(L+S)×e}`。
- RNN 解码器：双层 GRU/LSTM，输入为 `concat(v, emb(y_{t-1}))`，输出下一个标签分布 `p(y_t | v, y_{<t})`。
- 辅助多标签头（并行）：在 `v` 上接 `Linear(L) + Sigmoid` 直接回归多热标签，显式优化 `BCE` 以提升召回。

结构示意：`image -> CNN -> v -> {RNN seq head, BCE multi-hot head}`。

#### 4.2 训练目标

- 序列交叉熵：按 teacher forcing 计算 `CE(p_t, y_t)`，随机打乱同一图像内标签顺序（set-to-sequence）以降低顺序偏置；引入 coverage penalty 抑制重复。
- 多标签 BCE：对并行多热头计算 `BCE(σ(Wv), y)`；可使用 Focal Loss 替换或混合。
- 总损失：`L = α * L_BCE + β * L_CE + γ * L_cov`，推荐初始 `α=1.0, β=0.5, γ=0.1`，验证集调参。

#### 4.3 训练细节

- Teacher forcing 概率从 1.0 线性下降至 0.5；
- Optimizer：AdamW，lr=1e‑3（仅 RNN/头），CNN 冻结 5～10 epoch 后解冻，解冻后 lr=1e‑4；
- 批大小：32～128（AMP 混合精度）；
- 正则：Dropout=0.3（RNN 输入/隐层），Label Smoothing=0.05；
- 数据增强：随机裁剪、水平翻转、ColorJitter、弱高斯模糊；
- 早停与最佳权重：按 `mAP@elements` 或 `micro‑F1` 早停。

#### 4.4 推理与解码

- Gate：独立二分类头（在 `v` 上 `Linear(1)` + Sigmoid + BCE），或单独训练一个轻量 CNN 二分类器作为 A 模型。
- 多标签：
  - RNN：beam search（宽度 3～5），去重并忽略 `<eos>` 后得到集合；
  - 并行头：直接输出 `σ(Wv)`；
  - 融合（本路内部）：`s_unified = λ * σ(Wv) + (1‑λ) * s_rnn`（`λ` 验证集调优）。
- 阈值：逐类阈值 `τ_l` 在验证集按 `F1` 最大化或 Youden 指数标定。

---

### 5. finetuner 模块：CLIP/Gemma3 开放词表多标签

参考：`docs/literature/Open-Vocabulary_Multi-label_Image_Classification_with_Pretrained_Vision-Language_Model.pdf` 与 `docs/literature/Hybrid CNN-RNN + CLIP-Based Pipeline for Enhanced.md`

#### 5.1 模型选择

- 视觉编码器：OpenCLIP ViT‑B/16（起步，显存友好）或 ViT‑L/14（资源允许时）。
- 文本编码器：沿用 OpenCLIP 文本编码器；如需中文增强，可并行引入 Gemma3‑4b 文本侧产出的句向量，通过对齐层适配到同一嵌入维度（可选）。
- 参数高效微调：LoRA/Adapter/VPT（仅调视觉侧少量参数，文本侧多数冻结）。

#### 5.2 文本提示（Prompts）

- 多模板集成：
  - 中文模板：`"这是一张{要素}的气象图"`, `"包含{要素}要素的天气图"`
  - 英文模板：`"a meteorological chart showing {label}"`, `"weather map with {label}"`
- 每类取多模板编码并做均值/加权作为类原型 `t_l`；原型按温度参数 `T` 归一化。

#### 5.3 训练目标（多阳性对比 + 原型 BCE）

- 多阳性对比学习（multi‑positive contrastive）：对一个图像 `x`，其正集合 `P(x)` 为所有真标签对应的文本原型；损失鼓励 `sim(f(x), t_p)` 高于所有负类原型。
- 原型 BCE：对图像嵌入 `f(x)` 与所有类原型 `t_l` 的相似度经线性标定后过 Sigmoid，优化二元交叉熵；
- 总损失：`L = L_contrastive + η * L_BCE`（`η=0.1~0.3` 起步）。

#### 5.4 训练细节

- 冻结策略：先冻结视觉/文本编码器，仅训练原型/适配层与 LoRA，稳定后逐步解冻视觉后几层（lr 更小 1e‑5）。
- 优化器：AdamW，基 lr=1e‑4（头部），权重衰减 0.05；
- 批大小：256（可梯度累计）；
- 增强：CLIP 风格（随机裁剪、颜色抖动、Resize、随机遮挡少量）；
- 温度与标定：学习型温度参数 `T`；验证集做温度/阈值双重校准。

#### 5.5 零样本与开放词表

- 零样本基线：不训练直接用模板相似度评分，给出初始 `mAP`。 
- 扩展新类：仅需在 `labels.yaml` 中添加新类与模板，无需重新训练（或做少量增量微调）。

---

### 5A. 任务3：Gemma3 产品分类（图像 + 要素文本）

参考：`docs/literature/Open-Vocabulary_Multi-label_Image_Classification_with_Pretrained_Vision-Language_Model.pdf` 与 `docs/literature/Hybrid CNN-RNN + CLIP-Based Pipeline for Enhanced.md`

- 输入构造：
  - 图像：原图（含合成的标题/Logo 时保留）。
  - 要素文本：由任务2（CNN‑RNN Unified）输出的要素集合 S_elem 文本化，如“图中包含：风、气压、降水”。
  - 标题/Logo 提示：若样本被增强过，抽取标题关键短语与“带机构水印”等提示拼接进 prompt。
- 模型与微调：
  - 基座：Gemma3‑4b 多模态/图文指令微调。
  - 参数高效微调（LoRA/Adapter），优先文本/融合层；视觉侧仅末层解冻。
- 训练目标：
  - 指令式单标签分类，将产品分类视为受限生成；对正确产品名的生成概率最大化（序列交叉熵）。
  - 候选集来自 `docs/labels.yaml: products`，必要时按要素先筛后选（如包含 wind 则优先风场相关产品）。
  - 难负采样：同要素不同时效/不同高度层作为难负类。
  - 约束解码：推理时将输出限制在候选词表或采用指针式选择器。
- 训练细节：
  - lr_head=2e-4，lr_lora=1e-4，bs=32～64（梯度累计）；
  - 文本正则：要素顺序打乱、标题同义改写、单位/时效格式扰动（24小时/24h/24-hr）。
  - 评估：Top-1/Top-5、同要素产品间混淆矩阵。
- 可扩展性：
  - `labels.yaml` 维护 `products`（中文/英文/别名/缩写/单位与层次：850hPa/500hPa/2m 等）。
  - 新增产品时仅更新词表并做少量增量微调。
  - 当 S_elem 置信度低时，可标注“不确定”或仅用标题上下文降低误导。

### 6. 级联与跨路融合

令 `s_unified ∈ [0,1]^L` 为 CNN‑RNN 路线输出，`s_clip ∈ [0,1]^L` 为 CLIP 路线输出：

- 简单加权：`s = w1 * s_unified + w2 * s_clip`，`w1+w2=1`，在验证集网格搜索；
- 学习型融合：用验证集训练一层 `LogisticRegression`/`MLP` 以 `(s_unified, s_clip)` 为输入预测每一类的最终概率；
- 门控：当 `p_meteo < τ_gate`，直接输出非气象图，无要素；当 `p_meteo ≥ τ_gate`，对 `s` 做阈值化得到标签集合；
- 阈值：逐类阈值 `τ_l` 由验证集 `F1` 最大化获得；必要时使用 per‑class temperature scaling 或 logit adjustment 缓解长尾。

---

### 7. 评估指标与校准

- Gate：AUC、Accuracy、Precision/Recall、F1、EER；
- Elements：mAP（Pascal VOC / COCO 风格都可报告）、micro/macro‑F1、per‑class AP、Coverage Error；
- 校准：ECE、Brier Score；并输出逐类阈值表与温度参数。

---

### 8. 资源与工程落地

- 环境：CUDA 12.x，PyTorch ≥ 2.1；`environment.yml` 中加入 `open_clip_torch`, `timm`, `scikit-learn` 等依赖。
- 目录约定：
  - `trainer/`：CNN‑RNN 训练代码（数据集、模型、损失、训练脚本、推理与阈值导出）。
  - `finetuner/`：CLIP/Gemma3 微调代码（prompt 生成、原型缓存、对比学习、推理与标定）。
  - `docs/labels.yaml`：标签与同义词、模板。
  - `train_data/`：原始/增强数据与划分清单。
- 日志：`reports`/`wandb` 记录 mAP、F1、loss 曲线；保存最佳权重与阈值表 `thresholds.json`。

---

### 9. 建议的超参与起始配置（可按资源调整）

- CNN‑RNN：ResNet‑50 + 2×GRU(512)；img 256；bs=64；α/β/γ=1.0/0.5/0.1；lr_head=1e‑3, lr_backbone=1e‑4；epoch=50；beam=5；λ=0.5。
- CLIP 微调：OpenCLIP ViT‑B/16；LoRA r=8；bs=256（梯度累计 4×64）；lr_head=1e‑4, lr_lora=5e‑5；η=0.2；epoch=20；模板每类 4‑8 条；温度学习。
- 阈值：`τ_gate` 以 `EER` 或最大 `F1` 确定；逐类 `τ_l` 以 `F1` 最大化确定；必要时 per‑class temperature scaling。

---

### 10. 里程碑与产出物

1) Week 1‑2：数据清洗与标注格式统一，建立零样本 CLIP 基线，导出初始阈值表；
2) Week 3‑4：实现并训练 CNN‑RNN（含并行 BCE 头），得到 `mAP↑`、`F1↑` 基线；
3) Week 5：实现 CLIP 参数高效微调（LoRA/Adapter），完成温度与阈值校准；
4) Week 6：双路融合与线上 A/B，固化推理门控与阈值表，产出最终版本模型与部署说明；

---

### 11. 硬件环境建议（单卡 3060 Ti + 13600KF, PyTorch）

- 通用设置：
  - AMP 混合精度（fp16）与 GradScaler；`torch.backends.cuda.matmul.allow_tf32=True` 提升吞吐；
  - DataLoader：`pin_memory=True`、`persistent_workers=True`、`prefetch_factor=2`；`num_workers=8~12`；
  - 内存与算力：使用 `channels_last`，必要时开启 gradient checkpointing 与 `torch.compile`（PyTorch≥2.1）。

- 任务2（CNN‑RNN Unified）：
  - 主干：ResNet‑50；RNN：GRU×2，hidden=384~512；label emb=256；
  - 图像尺寸：224（优先），资源足够时 256；
  - 批大小（8GB 显存参考）：224→bs=32（不足时 16）；256→bs=16（`acc_steps=2` 等效 32）；
  - 训练：冻结主干 5~10 epoch；解冻后 lr_backbone=1e‑4；开启 AMP 与 checkpointing；
  - 推理：半精度 + `torch.inference_mode()`；beam width=3。

- 任务3（Gemma3 产品分类）：
  - 视觉塔：ViT‑B/16 冻结或仅末层解冻，分辨率 224；
  - 模型规模与微调：Gemma3‑4b + QLoRA（4‑bit）或 8‑bit，LoRA r=8, α=16，目标模块为注意力 Q,K,V,O；
  - 批大小与序列：global bs≈32（如 per‑GPU 4 × acc_steps 8），seq_len 256~512；
  - 训练技巧：AMP + 4‑bit（bitsandbytes）、gradient checkpointing、CPU offload（accelerate 可选）；
  - 文本裁剪：控制要素与标题长度，避免超长；候选先筛再选以降低计算；
  - 推理：4/8‑bit + 受限解码，视觉特征可离线缓存。

- 水印/标题增强实现：
  - 划分后合成并缓存到磁盘，记录元信息（标题、Logo 位置）；
  - 字体放置 `train_data/fonts/`，标题模板包含时间/单位关键字（“24小时”“850hPa”“2m”）。

- 依赖：
  - `bitsandbytes`, `peft`, `transformers`, `accelerate`, `timm`, `wandb`。

### 参考文献与材料（项目内文件）

- Wang, CNN‑RNN: A Unified Framework for Multi‑label Image Classification（`docs/literature/Wang_CNN-RNN_A_Unified_CVPR_2016_paper.pdf`）
- Open‑Vocabulary Multi‑label Image Classification with Pretrained Vision‑Language Model（`docs/literature/Open-Vocabulary_Multi-label_Image_Classification_with_Pretrained_Vision-Language_Model.pdf`）
- Hybrid CNN‑RNN + CLIP‑Based Pipeline（`docs/literature/Hybrid CNN-RNN + CLIP-Based Pipeline for Enhanced.md`）


