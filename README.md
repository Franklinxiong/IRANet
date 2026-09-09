
# IRANet: Illumination-Adaptive RGB-only Glass Detection

<p align="center">[English](./README.md) | [中文](./README.zh-CN.md)</p>

Glass is transparent and reflection-prone, which makes it one of the harder
things to segment from a single RGB image. Most existing glass detectors lean
on depth sensors, but that does not transfer well to a UAV platform: depth
cameras' size, weight, and power draw run against the SWaP constraints of small drones. This repository contains the
training, evaluation, and inference code for IRANet, a fully **RGB-only** glass
segmentation network designed for UAV footage captured at 5-30 m range under
strong illumination changes.

The model is a ResNeXt-101 encoder-decoder with three lightweight modules
inserted at every decode stage:

- **Illumination Invariant Module** - a squeeze-and-excitation style channel
  attention that suppresses illumination-sensitive channels while keeping the
  geometric structure channels intact.
- **Reflection Enhancement Module** - a residual branch with a learnable scalar
  (initialised to zero) that adaptively amplifies weak reflection features,
  without needing reflection annotations.
- **Enhanced Attention Module** - a channel + spatial attention block that
  focuses the decoder on glass boundaries and small regions.

Only the illumination module gives a clear gain on GDD; the other two tighten
boundaries but are roughly neutral in IoU/F1. See the ablation section below.

## Results

| Dataset | Split | IoU | F1 |
| --- | --- | --- | --- |
| GDD | test | 0.8758 | 0.9255 |
| Trans10K | easy | 0.9023 | 0.9486 |
| Trans10K | hard | 0.7520 | 0.8585 |

Ablation variants (no illumination / no reflection / no attention / full) live
in `ablation/`, with per-variant metrics saved under `ablation_results/`.

## Repository layout

```
.
|-- train_rgb_only.py          # baseline RGB-only training
|-- model_rgb_only.py          # baseline network (RGBOnlyGlassNet)
|-- optimize_rgb_only.py       # optimisation / tuning experiment script
|-- ablation_study.py          # cross-variant ablation driver
|-- batch_inference_test.py    # batch inference on a folder of images
|-- data_loaderd.py            # shared data loading utilities
|-- metrics.py / evaluation_fast.py / lr_scheduler.py / lovasz_losses.py / misc.py
|-- requirements.txt
|-- ablation/                  # ablation model variants + train/test scripts
|-- ablation_results/          # saved per-variant metrics and figures
|-- backbone/                  # ResNet / ResNeXt-101 backbones
|-- evaluation/                # GDD / GSD / depth eval scripts
|-- improved/                  # improved training, testing, and inference
|-- data/                      # training / test images (see below)
```

## Installation

The project is written in PyTorch and tested with torch 2.x on both macOS
(Apple Silicon) and Linux with CUDA. Create a fresh environment and install:

```bash
conda create -n glass_detection python=3.10
conda activate glass_detection
pip install -r requirements.txt
```

On GPU machines, install a CUDA build of PyTorch first, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`pydensecrf` is optional (used only for CRF post-processing); if it is missing
the scripts skip CRF refinement automatically.

## Dataset preparation

The two datasets are **not bundled** with this repository. Download them
yourself (or copy them from a local dataset) and arrange them as below.

### GDD

The GDD (Glass Detection Dataset) is obtained by request: head to the official
project page and apply via the Application Link, then the authors will grant
you access.

- Project page: https://mhaiyang.github.io/CVPR2020_GDNet/ (request via the project page)

### Trans10K

Trans10K (train / validation / test splits) can be downloaded from the official
project page, which provides Google Drive and Baidu Drive mirrors with
extraction codes, together with the official reference code:

- Project page: https://xieenze.github.io/projects/TransLAB/TransLAB.html
- Reference code: https://github.com/xieenze/Segment_Transparent_Objects

### Expected directory layout

The training code expects the dataset to be laid out as:

```
data/
|-- train/train/
|   |-- images/    # *.jpg RGB images
|   `-- masks/     # *.png binary masks (same base name or *_mask.png)
|-- validation/validation/
|   |-- easy/{images,masks}
|   `-- hard/{images,masks}
`-- test/test/
    |-- easy/{images,masks}
    `-- hard/{images,masks}
```

For the improved pipeline, `--split easy|hard|all` runs against the
`data/test/test/...` directories; `--dataset data_test` points the evaluator
at `data/test`. Trans10K results in the table above were produced with the same
split convention (`trans10k` preset).

## Training

Baseline model:

```bash
python train_rgb_only.py
```

Run configuration is set with constants at the top of the script (data
directory, image extensions, epoch count, batch size, learning rate, and
training resolution). The default checkpoint directory is
`saved_models/rgb_only_model/`.

Improved pipeline (warmup, cosine annealing, EMA, AMP, edge loss) on a 24 GB
GPU, batch 16 recommended:

```bash
python improved/train.py --batch_size 16 --epochs 150
```

Useful options: `--lr`, `--size`, `--device auto`, `--resume <ckpt>`,
`--no_amp`, `--no_auto_test`, `--keep_all_checkpoints`, and
`--backbone_path <path to resnext_101_32x4d.pth>`.

## Testing

Evaluate a trained model against a pre-defined split (easy / hard / all):

```bash
python improved/test.py --model_path improved/models/best.pth --split easy
python improved/test.py --model_path improved/models/best.pth --split hard
python improved/test.py --model_path improved/models/best.pth --split all
```

Test custom images/masks directories or a single image:

```bash
python improved/test.py --model_path improved/models/best.pth \
    --image_dir path/to/images --mask_dir path/to/masks
python improved/test.py --model_path improved/models/best.pth --single_img test.jpg
```

Results are written to `improved/results` by default.

## Inference

Run on a folder of images with the baseline flow:

```bash
python batch_inference_test.py \
    --input test_data/original --output test_data/detected \
    --model improved/models/best.pth
```

Or with the improved model:

```bash
python improved/inference.py --input PATH/TO/IMAGES --output improved/result
```

The scripts save JET heat-map overlays as JPG. Add `--no_cuda` to force CPU
and `--use_crf` to enable CRF refinement where available.

## Citation and license

If you use this code in your own work, please cite the paper once it is
publicly available. The repository is released under the BSD-3-Clause license.

