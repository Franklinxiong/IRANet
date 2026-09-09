# 消融实验 (Ablation Study)

对 RGB-only 玻璃检测模型中的三个核心模块进行消融实验，评估每个模块的贡献。

## 三个被评估模块

| 模块 | 文件中的类名 | 功能 |
|------|-------------|------|
| **IlluminationInvariantModule** | `光照不变特征提取模块` | 针对弱光条件提取光照不变特征 |
| **ReflectionEnhancementModule** | `反射特征增强模块` | 针对低反射玻璃增强反射特征 |
| **EnhancedAttentionModule** | `增强的注意力机制模块` | 通道+空间注意力提高特征选择能力 |

## 文件结构

```
ablation/
├── __init__.py               # 模块初始化
├── README.md                 # 本文件
├── models.py                 # 四个模型变体定义
├── train_ablation.py         # 统一训练脚本
├── test_ablation.py          # 统一测试脚本
└── checkpoints/              # 训练生成的权重（自动创建）
    ├── no_illumination/
    ├── no_reflection/
    ├── no_attention/
    ├── no_all/
    └── full/
```

## 模型变体

| 变体名 | 类 | 说明 |
|--------|-----|------|
| `full` | `RGBOnlyGlassNet` | 完整模型（对照基准） |
| `no_illumination` | `RGBOnlyGlassNet_NoIllumination` | 移除光照不变模块 |
| `no_reflection` | `RGBOnlyGlassNet_NoReflection` | 移除反射增强模块 |
| `no_attention` | `RGBOnlyGlassNet_NoAttention` | 移除增强注意力模块 |
| `no_all` | `RGBOnlyGlassNet_NoAll` | 移除所有三个模块（消融基线） |

## 使用方式

### 1. 训练

训练单个变体：
```bash
# 训练（移除光照不变模块）
python ablation/train_ablation.py --variant no_illumination

# 训练（移除反射增强模块）
python ablation/train_ablation.py --variant no_reflection

# 训练（移除增强注意力模块）
python ablation/train_ablation.py --variant no_attention

# 训练基线模型（移除所有模块）
python ablation/train_ablation.py --variant no_all

# 训练完整模型（用于对照）
python ablation/train_ablation.py --variant full
```

参数说明：
- `--variant`：模型变体（必选）
- `--epochs`：训练轮数，默认200
- `--batch_size`：批次大小，默认14
- `--lr`：学习率，默认1e-4

模型权重自动保存到 `ablation/checkpoints/{variant}/` 目录。

### 2. 测试

测试单个变体：
```bash
python ablation/test_ablation.py --variant no_illumination
python ablation/test_ablation.py --variant no_reflection
python ablation/test_ablation.py --variant no_attention
python ablation/test_ablation.py --variant no_all
python ablation/test_ablation.py --variant full
```

测试所有已训练的变体并生成对比报告：
```bash
python ablation/test_ablation.py --variant all
```

参数说明：
- `--variant`：要测试的变体（必选），`all`=测试所有变体并对比
- `--checkpoint`：自定义权重路径
- `--output_dir`：输出目录，默认 `ablation_results/`
- `--batch_size`：测试批次大小，默认4

### 3. 输出结果

测试脚本的输出全部保存到 `ablation_results/` 目录：

```
ablation_results/
├── no_illumination/
│   ├── easy_results.png         # easy测试集多阈值柱状图
│   ├── easy_metrics.json         # easy测试集详细指标
│   ├── hard_results.png          # hard测试集多阈值柱状图
│   └── hard_metrics.json         # hard测试集详细指标
├── no_reflection/
│   └── ...
├── no_attention/
│   └── ...
├── no_all/
│   └── ...
├── full/
│   └── ...
└── comparison/                    # 对比分析（仅 --variant all 时生成）
    ├── comparison_report.txt      # 对比文本报告
    ├── comparison_easy.png        # easy测试集对比柱状图
    ├── comparison_hard.png        # hard测试集对比柱状图
    ├── ablation_radar.png         # 模块性能雷达图
    └── ablation_summary.json      # 所有变体的汇总指标
```

## 评估指标

| 指标 | 说明 |
|------|------|
| F1分数 | 精确率和召回率的调和平均 |
| IoU | 预测与标签的交并比 |
| 精确率(Precision) | 预测为正的样本中实际为正的比例 |
| 召回率(Recall) | 实际为正的样本中被正确预测的比例 |
| Dice系数 | 与IoU类似的集合相似度指标 |
| MAE | 平均绝对误差，越低越好 |
| BER | Balanced Error Rate，越低越好 |
| 准确率(Accuracy) | 正确预测的样本比例 |

测试时在 3 个阈值 (0.3, 0.5, 0.7) 下分别计算指标，方便观察模型在不同置信度下的表现。
