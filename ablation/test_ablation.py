"""
Test script for the ablation variants.

Evaluates every variant on the easy/hard splits and writes to
ablation_results/:
  1. per-variant metrics on easy and hard
  2. a bar chart comparing all variants
  3. a radar chart
  4. a text report with a per-module importance analysis

Usage:
    python ablation/test_ablation.py --variant baseline
    python ablation/test_ablation.py --variant plus_attention
    python ablation/test_ablation.py --variant all   # every trained variant
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import glob
from PIL import Image
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# put the project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation.models import MODEL_VARIANTS


# ============================================================================
# dataset
# ============================================================================

class ToTensorLab(object):
    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        if label.dtype != np.float32:
            label = label.astype(np.float32)

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
        image = image / np.max(image)

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        if np.max(label) < 1e-6:
            tmpLbl = label
        else:
            tmpLbl = label / np.max(label)

        tmpImg = tmpImg.transpose((2, 0, 1))
        if len(tmpLbl.shape) == 2:
            tmpLbl = tmpLbl[np.newaxis, :, :]

        return {'image': torch.from_numpy(tmpImg).float(), 'label': torch.from_numpy(tmpLbl).float()}


class GlassTestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(
            glob.glob(os.path.join(image_dir, '*.jpg')) +
            glob.glob(os.path.join(image_dir, '*.png'))
        )
        print(f"  找到 {len(self.image_files)} 个测试图像")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(img_name)[0]

        image = Image.open(img_path).convert('RGB')
        image = image.resize((512, 512), Image.BILINEAR)
        image = np.array(image)

        mask_path = None
        mask_candidates = [
            os.path.join(self.mask_dir, f"{name_without_ext}_mask.png"),
            os.path.join(self.mask_dir, f"{name_without_ext}.png"),
            os.path.join(self.mask_dir, f"{name_without_ext}_mask.jpg"),
            os.path.join(self.mask_dir, f"{name_without_ext}.jpg"),
        ]
        for c in mask_candidates:
            if os.path.exists(c):
                mask_path = c
                break

        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((512, 512), Image.NEAREST)
            mask = np.array(mask)
        else:
            print(f"  警告: {img_name} 没有对应掩码")
            mask = np.zeros((512, 512), dtype=np.uint8)

        if mask.max() > 1:
            mask = (mask > 30).astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        if self.transform:
            transformed = self.transform({'image': image, 'label': mask})
            image = transformed['image']
            mask = transformed['label']
        else:
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {'image': image, 'label': mask, 'name': img_name}


# ============================================================================
# metrics
# ============================================================================

def compute_metrics(predictions, targets, threshold=0.5):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    pred_binary = (pred_flat > threshold).astype(np.int32)
    target_binary = (target_flat > 0.5).astype(np.int32)

    eps = 1e-7
    tp = np.sum((pred_binary == 1) & (target_binary == 1))
    tn = np.sum((pred_binary == 0) & (target_binary == 0))
    fp = np.sum((pred_binary == 1) & (target_binary == 0))
    fn = np.sum((pred_binary == 0) & (target_binary == 1))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    mae = np.mean(np.abs(pred_flat - target_flat))
    ber = 100 * (1 - 0.5 * (tp / (tp + fn + eps) + tn / (tn + fp + eps)))

    return {
        'precision': float(precision), 'recall': float(recall),
        'f1_score': float(f1), 'iou': float(iou),
        'accuracy': float(accuracy), 'dice': float(dice),
        'mae': float(mae), 'ber': float(ber),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def evaluate_model(model, dataloader, device, model_name=""):
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, labels, names = (
                batch['image'].to(device),
                batch['label'].to(device),
                batch['name']
            )
            outputs = model(images)
            # use the first prediction head (layer1_predict) as the final output
            predictions = torch.sigmoid(outputs[0])

            for i in range(predictions.shape[0]):
                pred = predictions[i:i+1]
                label = labels[i:i+1]
                metrics = compute_metrics(pred, label)
                metrics['name'] = names[i]
                all_metrics.append(metrics)

            if (batch_idx + 1) % 10 == 0:
                print(f"    已处理 {batch_idx + 1}/{len(dataloader)} 批次")

    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            if key not in ['name', 'tp', 'tn', 'fp', 'fn']:
                values = [m[key] for m in all_metrics]
                avg_metrics[key] = float(np.mean(values))
        avg_metrics['total_tp'] = sum(m['tp'] for m in all_metrics)
        avg_metrics['total_tn'] = sum(m['tn'] for m in all_metrics)
        avg_metrics['total_fp'] = sum(m['fp'] for m in all_metrics)
        avg_metrics['total_fn'] = sum(m['fn'] for m in all_metrics)

    return avg_metrics, all_metrics


# ============================================================================
# visualization
# ============================================================================

def plot_comparison_bar(results, output_path, title="消融实验对比"):
    """Draw a bar chart comparing all variants."""
    variants = list(results.keys())
    metrics_to_plot = ['f1_score', 'iou', 'precision', 'recall', 'dice', 'mae', 'ber']
    metric_names_map = ['F1分数', 'IoU', '精确率', '召回率', 'Dice系数', 'MAE', 'BER(%)']
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

    # transpose the data structure
    metric_values = {}
    for metric in metrics_to_plot:
        metric_values[metric] = [results[v].get(metric, 0) for v in variants]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names_map)):
        ax = axes[idx // 4, idx % 4]
        values = metric_values[metric]
        bars = ax.bar(variants, values, color=colors[:len(variants)])
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=30)

        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + max(max(values) * 0.02, 0.01),
                f'{v:.3f}', ha='center', va='bottom', fontsize=8
            )

        if metric not in ['mae', 'ber']:
            ax.set_ylim(0, max(max(values) * 1.15, 0.5))
        else:
            ax.set_ylim(0, max(values) * 1.3)

    for idx in range(len(metrics_to_plot), 8):
        axes[idx // 4, idx % 4].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_radar(results, output_path, title="消融实验雷达图"):
    """Draw a radar chart comparing all variants."""
    variants = list(results.keys())
    metrics_to_plot = ['f1_score', 'iou', 'precision', 'recall', 'dice']
    metric_names = ['F1分数', 'IoU', '精确率', '召回率', 'Dice系数']
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')

    for i, variant in enumerate(variants):
        values = [results[variant].get(m, 0) for m in metrics_to_plot]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=variant, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='消融实验测试脚本（从baseline逐个添加模块）')
    parser.add_argument('--variant', type=str, required=True,
                       choices=['baseline', 'plus_illumination', 'plus_reflection',
                                'plus_attention', 'all'],
                       help='要测试的模型变体；all=测试所有已训练的变体')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型权重路径（默认使用 ablation/checkpoints/{variant}/{variant}_best.pth）')
    parser.add_argument('--test_easy_dir', type=str, default='data/test/test/easy')
    parser.add_argument('--test_hard_dir', type=str, default='data/test/test/hard')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='ablation_results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu', 'mps'])
    args = parser.parse_args()

    # - set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"使用CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("使用MPS (Mac)")
        else:
            device = torch.device('cpu')
            print("使用CPU")
    else:
        device = torch.device(args.device)

    # - decide which variants to evaluate
    # short display names used on the charts
    variant_display = {
        'baseline':          'Baseline',
        'plus_illumination': '+Illumination',
        'plus_reflection':   '+Reflection',
        'plus_attention':    '+Attention (Full)',
    }

    # ordered per the ablation pipeline
    variant_order = ['baseline', 'plus_illumination', 'plus_reflection', 'plus_attention']

    if args.variant == 'all':
        # auto-detect which variants have trained weights
        variant_list = []
        for v in variant_order:
            # look for best.pth or final.pth
            ckpt_dir = os.path.join('ablation', 'checkpoints', v)
            if os.path.exists(ckpt_dir):
                pth_files = glob.glob(os.path.join(ckpt_dir, '*.pth'))
                if pth_files:
                    variant_list.append(v)
        if not variant_list:
            print("错误: 没有找到任何已训练的变体。请先运行训练脚本。")
            return
    else:
        variant_list = [args.variant]

    print(f"\n将测试 {len(variant_list)} 个变体: {[variant_display.get(v, v) for v in variant_list]}")

    # - check the test directory
    test_sets = []
    for name, test_dir in [('easy', args.test_easy_dir), ('hard', args.test_hard_dir)]:
        image_dir = os.path.join(test_dir, 'images')
        mask_dir = os.path.join(test_dir, 'masks')
        if os.path.exists(image_dir) and os.path.exists(mask_dir):
            test_sets.append((name, test_dir))
            print(f"找到测试集: {name}")
        else:
            print(f"警告: 测试集 {test_dir} 不完整")

    if not test_sets:
        print("错误: 没有找到有效的测试集")
        return

    # data transform
    transform = ToTensorLab(flag=0)

    # create the output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # results live here: {test_name: {variant: avg_metrics}}
    all_results = {ts[0]: {} for ts in test_sets}

    # - evaluate each variant
    for variant_key in variant_list:
        display_name = variant_display.get(variant_key, variant_key)
        print(f"\n{'='*60}")
        print(f"评估变体: {display_name} ({variant_key})")
        print(f"{'='*60}")

        # resolve the weight path
        if args.checkpoint and args.variant != 'all':
            checkpoint_path = args.checkpoint
        else:
            ckpt_dir = os.path.join('ablation', 'checkpoints', variant_key)
            # prefer best.pth, fall back to final.pth, then the newest checkpoint
            best_path = os.path.join(ckpt_dir, f'{variant_key}_best.pth')
            final_path = os.path.join(ckpt_dir, f'{variant_key}_final.pth')
            if os.path.exists(best_path):
                checkpoint_path = best_path
            elif os.path.exists(final_path):
                checkpoint_path = final_path
            else:
                pth_files = glob.glob(os.path.join(ckpt_dir, '*.pth'))
                if pth_files:
                    checkpoint_path = max(pth_files, key=os.path.getmtime)
                    print(f"  使用最新检查点: {checkpoint_path}")
                else:
                    print(f"  错误: {ckpt_dir} 中没有 .pth 文件")
                    continue

        if not os.path.exists(checkpoint_path):
            print(f"  错误: 权重文件不存在: {checkpoint_path}")
            continue

        print(f"  权重: {checkpoint_path}")

        # - load the model
        try:
            model_class = MODEL_VARIANTS[variant_key]
            model = model_class(backbone_path=None)
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device)
            print(f"  模型加载成功 ({type(model).__name__})")
        except Exception as e:
            print(f"  模型加载失败: {e}")
            continue

        # - output directory for this variant
        variant_output_dir = os.path.join(output_dir, variant_key)
        os.makedirs(variant_output_dir, exist_ok=True)

        # - evaluate on each test set
        for test_name, test_dir in test_sets:
            print(f"\n  --- 测试集: {test_name} ---")
            image_dir = os.path.join(test_dir, 'images')
            mask_dir = os.path.join(test_dir, 'masks')

            dataset = GlassTestDataset(image_dir, mask_dir, transform=transform)
            dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)

            # evaluate at a 0.5 threshold
            avg_metrics, _ = evaluate_model(model, dataloader, device, f"{display_name}@{test_name}")

            # save the results
            all_results[test_name][variant_key] = avg_metrics

            # print
            print(f"  >> {display_name} @ {test_name}:")
            print(f"     F1={avg_metrics.get('f1_score', 0):.4f}, "
                  f"IoU={avg_metrics.get('iou', 0):.4f}, "
                  f"P={avg_metrics.get('precision', 0):.4f}, "
                  f"R={avg_metrics.get('recall', 0):.4f}")
            print(f"     Dice={avg_metrics.get('dice', 0):.4f}, "
                  f"MAE={avg_metrics.get('mae', 0):.4f}, "
                  f"BER={avg_metrics.get('ber', 0):.2f}%, "
                  f"Acc={avg_metrics.get('accuracy', 0):.4f}")

            # save the JSON
            json_path = os.path.join(variant_output_dir, f'{test_name}_metrics.json')
            with open(json_path, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            print(f"  指标已保存: {json_path}")

    # - build the comparison report (when more than one variant was tested)
    if len(variant_list) > 1:
        comparison_dir = os.path.join(output_dir, 'comparison')
        os.makedirs(comparison_dir, exist_ok=True)

        # bar chart
        for test_name in [ts[0] for ts in test_sets]:
            if all_results[test_name]:
                bar_path = os.path.join(comparison_dir, f'comparison_{test_name}.png')
                # label the charts with the short names
                display_results = {}
                for v in variant_list:
                    if v in all_results[test_name]:
                        display_results[variant_display.get(v, v)] = all_results[test_name][v]
                plot_comparison_bar(display_results, bar_path,
                                   f"{test_name}测试集 - 消融实验对比")
                print(f"对比图已保存: {bar_path}")

        # radar chart averaging easy and hard
        avg_results = {}
        for variant_key in variant_list:
            easy_m = all_results.get('easy', {}).get(variant_key, {})
            hard_m = all_results.get('hard', {}).get(variant_key, {})
            if easy_m and hard_m:
                d_name = variant_display.get(variant_key, variant_key)
                avg_results[d_name] = {}
                for metric in easy_m:
                    if metric not in ['total_tp', 'total_tn', 'total_fp', 'total_fn']:
                        avg_results[d_name][metric] = (
                            easy_m.get(metric, 0) + hard_m.get(metric, 0)
                        ) / 2

        if len(avg_results) > 1:
            radar_path = os.path.join(comparison_dir, 'ablation_radar.png')
            plot_radar(avg_results, radar_path, "消融实验 - 平均性能雷达图")
            print(f"雷达图已保存: {radar_path}")

        # == text report ==
        report_path = os.path.join(comparison_dir, 'comparison_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RGB-only玻璃检测模型消融实验对比报告\n")
            f.write("消融方向：从baseline逐个添加模块\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"设备: {device}\n")
            f.write(f"流水线顺序: rgb_layer → illumination → reflection → attention → contrast\n\n")

            # variant descriptions
            f.write("变体对照:\n")
            for v in variant_order:
                f.write(f"  {variant_display.get(v, v):25s} - {get_variant_description(v)}\n")
            f.write("\n")

            # result table per test set
            for test_name in [ts[0] for ts in test_sets]:
                f.write(f"{'='*80}\n")
                f.write(f"测试集: {test_name}\n")
                f.write(f"{'='*80}\n\n")

                # table header
                header = (f"{'变体':<25} {'F1分数':<10} {'IoU':<10} "
                         f"{'精确率':<10} {'召回率':<10} {'Dice':<10} "
                         f"{'MAE↓':<10} {'BER(%)↓':<10}")
                f.write(header + "\n")
                f.write("-" * 95 + "\n")

                best_f1 = -1
                best_v = ""
                for variant_key in variant_order:
                    if variant_key in all_results[test_name]:
                        m = all_results[test_name][variant_key]
                        d_name = variant_display[variant_key]
                        f.write(f"{d_name:<25} ")
                        f.write(f"{m.get('f1_score', 0):<10.4f} ")
                        f.write(f"{m.get('iou', 0):<10.4f} ")
                        f.write(f"{m.get('precision', 0):<10.4f} ")
                        f.write(f"{m.get('recall', 0):<10.4f} ")
                        f.write(f"{m.get('dice', 0):<10.4f} ")
                        f.write(f"{m.get('mae', 0):<10.4f} ")
                        f.write(f"{m.get('ber', 0):<10.2f}\n")
                        if m.get('f1_score', 0) > best_f1:
                            best_f1 = m.get('f1_score', 0)
                            best_v = d_name

                f.write(f"\n  [最佳] {best_v}: F1={best_f1:.4f}\n\n")

                # per-module gain analysis
                f.write("  --- 模块增益分析（与前一变体对比） ---\n")
                prev_metrics = None
                prev_name = ""
                for variant_key in variant_order:
                    if variant_key in all_results[test_name]:
                        m = all_results[test_name][variant_key]
                        d_name = variant_display[variant_key]
                        if prev_metrics is not None:
                            f1_gain = m.get('f1_score', 0) - prev_metrics.get('f1_score', 0)
                            iou_gain = m.get('iou', 0) - prev_metrics.get('iou', 0)
                            mae_change = m.get('mae', 0) - prev_metrics.get('mae', 0)
                            f.write(f"    {prev_name:25s} → {d_name:<25s}: "
                                   f"F1={f1_gain:+.4f}, IoU={iou_gain:+.4f}, MAE={mae_change:+.4f}\n")
                        prev_metrics = m
                        prev_name = d_name
                f.write("\n")

                # module importance, ranked by the drop relative to the full model
                if 'plus_attention' in all_results[test_name]:
                    full_f1 = all_results[test_name]['plus_attention'].get('f1_score', 0)
                    f.write("  --- 每个模块的重要性（移除该模块后的性能下降） ---\n")
                    # contribution of a module = full model minus the variant without it
                    # note: "removed" here means the variant that lacks the module
                    contributions = []
                    # baseline vs full gives the combined contribution of all three modules
                    if 'baseline' in all_results[test_name]:
                        base_f1 = all_results[test_name]['baseline'].get('f1_score', 0)
                        contributions.append(('三个模块合计', full_f1 - base_f1))
                    # plus_illumination has only illumination; missing reflection and attention
                    if 'plus_illumination' in all_results[test_name]:
                        ill_f1 = all_results[test_name]['plus_illumination'].get('f1_score', 0)
                    # plus_reflection has illumination and reflection; missing attention
                    if 'plus_reflection' in all_results[test_name]:
                        ref_f1 = all_results[test_name]['plus_reflection'].get('f1_score', 0)
                        contributions.append(('增强注意力', full_f1 - ref_f1))
                        contributions.append(('反射增强', ref_f1 - ill_f1))
                        contributions.append(('光照不变', ill_f1 - base_f1))

                    contributions.sort(key=lambda x: x[1], reverse=True)
                    for name, contrib in contributions:
                        pct = contrib / full_f1 * 100 if full_f1 > 0 else 0
                        f.write(f"    {name:<20s}: F1提升 {contrib:.4f} ({pct:+.1f}%)\n")
                    f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("实验完成\n")
            f.write("=" * 80 + "\n")

        print(f"\n对比报告已保存: {report_path}")

        # summary JSON
        summary_path = os.path.join(output_dir, 'ablation_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"汇总结果已保存: {summary_path}")

        # print the summary
        print(f"\n{'='*60}")
        print("消融实验摘要")
        print(f"{'='*60}")
        for test_name in [ts[0] for ts in test_sets]:
            print(f"\n{test_name}测试集:")
            for variant_key in variant_order:
                if variant_key in all_results[test_name]:
                    m = all_results[test_name][variant_key]
                    print(f"  {variant_display[variant_key]:<25s}: "
                          f"F1={m.get('f1_score', 0):.4f}, IoU={m.get('iou', 0):.4f}")

    print(f"\n{'='*60}")
    print("消融实验测试完成!")
    print(f"{'='*60}")


def get_variant_description(variant):
    descriptions = {
        'baseline':          '无三个模块（仅 backbone + cr + contrast + up）',
        'plus_illumination': '添加光照不变模块',
        'plus_reflection':   '添加反射增强模块',
        'plus_attention':    '添加增强注意力模块（完整模型）',
    }
    return descriptions.get(variant, variant)


if __name__ == '__main__':
    main()