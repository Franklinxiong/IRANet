# IRANet: 光照自适应的纯RGB玻璃检测（Illumination-Adaptive RGB-only Glass Detection）

<p align="center">[English](./README.md) | [中文](./README.zh-CN.md)</p>

玻璃是透明且容易反光的，这让它成为单张 RGB 图像分割中比较棘手的一类目标。现有的玻璃检测方法大多依赖深度传感器，但这一方案放到无人机平台上并不合适：深度相机的体积、重量和功耗都难以满足小型无人机的 SWaP（尺寸、重量与功耗）约束。本仓库收录了 IRANet 的训练、评测与推理代码。IRANet 是一个完全基于 **RGB-only**（仅 RGB）输入的玻璃分割网络，主要面向在 5-30 米距离、光照剧烈变化条件下拍摄的无人机影像。

模型的整体结构是 ResNeXt-101 编码器-解码器（encoder-decoder），在每个解码阶段插入三个轻量模块：

- **Illumination Invariant Module（光照不变模块）**：采用 squeeze-and-excitation 风格的通道注意力，压制对光照敏感的通道，同时保留几何结构通道。
- **Reflection Enhancement Module（反射增强模块）**：残差分支配合一个可学习的标量（初始化为 0），自适应放大微弱的反射特征，且不需要反射标注。
- **Enhanced Attention Module（增强注意力模块）**：通道 + 空间注意力块，让解码器更关注玻璃边界和小尺寸区域。

在 GDD 上只有光照模块带来了明显的收益，另外两个模块主要让边界更锐利，在 IoU/F1 上大致持平。详见下方的消融实验部分。

## 实验结果（Results）

| 数据集（Dataset） | 划分（Split） | IoU | F1 |
| --- | --- | --- | --- |
| GDD | test | 0.8758 | 0.9255 |
| Trans10K | easy | 0.9023 | 0.9486 |
| Trans10K | hard | 0.7520 | 0.8585 |

消融变体（去掉光照模块 / 去掉反射模块 / 去掉注意力模块 / 完整模型）的代码在 `ablation/` 中，各变体的指标保存在 `ablation_results/` 下。

## 目录结构（Repository layout）

```
.
|-- train_rgb_only.py          # 基础版 RGB-only 训练脚本
|-- model_rgb_only.py          # 基础网络（RGBOnlyGlassNet）
|-- optimize_rgb_only.py       # 调参 / 实验脚本
|-- ablation_study.py          # 跨变体消融实验入口
|-- batch_inference_test.py    # 对图片目录做批量推理
|-- data_loaderd.py            # 共用的数据加载工具
|-- metrics.py / evaluation_fast.py / lr_scheduler.py / lovasz_losses.py / misc.py
|-- requirements.txt
|-- ablation/                  # 消融模型变体及训练 / 测试脚本
|-- ablation_results/          # 各变体指标与图表
|-- backbone/                  # ResNet / ResNeXt-101 骨干网络
|-- evaluation/                # GDD / GSD / depth 评测脚本
|-- improved/                  # 改进版的训练、测试与推理
|-- data/                      # 训练 / 测试图片（见下方说明）
```

## 依赖安装（Installation）

项目基于 PyTorch 编写，已在 macOS（Apple Silicon）和带 CUDA 的 Linux 上用 torch 2.x 测试通过。建议先新建一个干净的环境再安装：

```bash
conda create -n glass_detection python=3.10
conda activate glass_detection
pip install -r requirements.txt
```

在 GPU 机器上，建议先安装 CUDA 版本的 PyTorch，例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`pydensecrf` 是可选依赖（仅用于 CRF 后处理）；如果未安装，脚本会自动跳过 CRF 细化步骤。

## 数据集准备（Dataset preparation）

本仓库**不包含**数据集，请自行下载（或复制本地已有的数据集）并按下面的结构摆放。

### GDD

GDD（玻璃检测数据集 Glass Detection Dataset）通过申请获取：访问官方项目页，点击其中的 Application Link 提交申请，作者审核后会提供下载。

- 项目页：https://mhaiyang.github.io/CVPR2020_GDNet/ （通过项目页申请数据集）

### Trans10K

Trans10K（含 train / validation / test 划分）可从官方项目页直接下载，页面提供 Google Drive 与百度网盘两种镜像及提取码，并附有官方配套代码：

- 项目页：https://xieenze.github.io/projects/TransLAB/TransLAB.html
- 官方代码仓库：https://github.com/xieenze/Segment_Transparent_Objects

### 目录结构说明

训练代码期望数据集按以下结构摆放：

```
data/
|-- train/train/
|   |-- images/    # *.jpg RGB 图像
|   `-- masks/     # *.png 二值掩码（与图像同名或 *_mask.png）
|-- validation/validation/
|   |-- easy/{images,masks}
|   `-- hard/{images,masks}
`-- test/test/
    |-- easy/{images,masks}
    `-- hard/{images,masks}
```

改进版流程里，`--split easy|hard|all` 会读取 `data/test/test/...` 目录；`--dataset data_test` 会让评测指向 `data/test`。上表中 Trans10K 的结果用的是相同的划分约定（`trans10k` 预设）。

## 训练（Training）

基础版模型：

```bash
python train_rgb_only.py
```

运行配置通过脚本顶部的常量设置（数据目录、图像扩展名、epoch 数、batch size、学习率、训练分辨率等），默认的权重保存目录是 `saved_models/rgb_only_model/`。

改进版流程（warmup、cosine 退火、EMA、AMP、边缘损失），在 24GB 显存的 GPU 上建议 batch_size 16：

```bash
python improved/train.py --batch_size 16 --epochs 150
```

常用参数：`--lr`、`--size`、`--device auto`、`--resume <ckpt>`、`--no_amp`、`--no_auto_test`、`--keep_all_checkpoints`、`--backbone_path <path to resnext_101_32x4d.pth>`。

## 测试（Testing）

按预设划分评测训练好的模型（easy / hard / all）：

```bash
python improved/test.py --model_path improved/models/best.pth --split easy
python improved/test.py --model_path improved/models/best.pth --split hard
python improved/test.py --model_path improved/models/best.pth --split all
```

对自定义的图片/掩码目录或单张图片进行测试：

```bash
python improved/test.py --model_path improved/models/best.pth \
    --image_dir path/to/images --mask_dir path/to/masks
python improved/test.py --model_path improved/models/best.pth --single_img test.jpg
```

结果默认写入 `improved/results`。

## 推理（Inference）

对图片目录用基础版流程批量推理：

```bash
python batch_inference_test.py \
    --input test_data/original --output test_data/detected \
    --model improved/models/best.pth
```

或者用改进版模型：

```bash
python improved/inference.py --input PATH/TO/IMAGES --output improved/result
```

脚本默认保存 JET 热力图叠加后的 JPG。加 `--no_cuda` 强制使用 CPU，加 `--use_crf` 在可用时启用 CRF 细化。

## 引用与许可（Citation and license）

如果这份代码对你的工作有帮助，请在论文公开后引用它。本仓库以 BSD-3-Clause 许可发布。

