"""
Improved test / inference script.

Quick test on a split:
python improved/test.py --model_path improved/models/best.pth --split easy
python improved/test.py --model_path improved/models/best.pth --split hard
python improved/test.py --model_path improved/models/best.pth --split all

Test the data/test datasets:
python improved/test.py --model_path improved/models/best.pth --split all --dataset data_test

Custom directory:
python improved/test.py --model_path improved/models/best.pth --image_dir path/to/images --mask_dir path/to/masks

Single image:
python improved/test.py --model_path improved/models/best.pth --single_img test.jpg

Full batch inference (saves heatmap overlays):
python batch_inference_test.py --input test_data/original --output test_data/detected_improved --model improved/models/best.pth
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import glob
import json

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improved.model import ImprovedGlassNet



# preset test split paths

TEST_DATASETS = {
    'trans10k': {
        'easy':   ('Trans10K/test/easy/images',   'Trans10K/test/easy/masks'),
        'hard':   ('Trans10K/test/hard/images',   'Trans10K/test/hard/masks'),
    },
    'data_test': {
        'easy':   ('data/test/test/easy/images',   'data/test/test/easy/masks'),
        'hard':   ('data/test/test/hard/images',   'data/test/test/hard/masks'),
    },
}


class TestDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, size=384):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')) +
                             glob.glob(os.path.join(image_dir, '*.png')))
        print(f"找到 {len(self.images)} 个测试图像")

    def __len__(self):
        return len(self.images)

    def _load_mask(self, img_path):
        if not self.mask_dir:
            return None
        name = os.path.splitext(os.path.basename(img_path))[0]
        for suffix in ['_mask.png', '.png']:
            p = os.path.join(self.mask_dir, f"{name}{suffix}")
            if os.path.exists(p):
                return np.array(Image.open(p).convert('L'))
        return None

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        orig_h, orig_w = image.size[1], image.size[0]
        image = image.resize((self.size, self.size), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        result = {'image': image, 'name': os.path.basename(img_path), 'orig_hw': (orig_h, orig_w)}
        mask = self._load_mask(img_path)
        if mask is not None:
            mask = np.array(Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST))
            mask = (mask > 30).astype(np.float32) if mask.max() > 1 else mask.astype(np.float32)
            result['label'] = torch.from_numpy(mask).unsqueeze(0).float()
        return result


@torch.no_grad()
def predict_single(model, img_path, device, size=384):
    model.eval()
    image = Image.open(img_path).convert('RGB')
    orig_h, orig_w = image.size[1], image.size[0]

    image_np = np.array(image.resize((size, size), Image.BILINEAR)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_tensor = torch.from_numpy(((image_np - mean) / std).transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    prob = torch.sigmoid(model(image_tensor)[0]).cpu().numpy()[0, 0]
    prob = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)) / 255.0
    return prob


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_metrics = []
    for batch in dataloader:
        images = batch['image'].to(device)
        outputs = model(images)
        prob = torch.sigmoid(outputs[0]).cpu().numpy()

        if 'label' in batch:
            gt = batch['label'].cpu().numpy()
            for i in range(prob.shape[0]):
                p, g = prob[i, 0].flatten(), gt[i, 0].flatten()
                pb, gb = p > 0.5, g > 0.5
                tp, fp, fn = (pb & gb).sum(), (pb & ~gb).sum(), (~pb & gb).sum()
                eps = 1e-7
                all_metrics.append({
                    'name': batch['name'][i],
                    'iou': (tp / (tp + fp + fn + eps)).item(),
                    'f1': (2 * tp / (2 * tp + fp + fn + eps)).item(),
                    'mae': np.abs(p - g).mean().item(),
                })
    return all_metrics



# quick-test helper


def run_split_test(model, device, split_name, dataset_name, args):
    """
    Run the tests for a single split.
    """
    img_dir, mask_dir = TEST_DATASETS[dataset_name][split_name]
    print(f"\n{'='*60}")
    print(f"测试 {dataset_name}/{split_name}")
    print(f"  图像: {img_dir}")
    print(f"  真值: {mask_dir}")
    print(f"{'='*60}")

    dataset = TestDataset(img_dir, mask_dir, args.size)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    metrics = evaluate(model, loader, device)

    if metrics:
        avg = {k: float(np.mean([m[k] for m in metrics])) for k in ['iou', 'f1', 'mae']}
        print(f"\n结果 ({len(metrics)} 样本):")
        print(f"  F1  = {avg['f1']:.4f}")
        print(f"  IoU = {avg['iou']:.4f}")
        print(f"  MAE = {avg['mae']:.4f}")

        # save the result
        out_dir = os.path.join(args.output, dataset_name, split_name)
        os.makedirs(out_dir, exist_ok=True)
        json.dump(avg, open(os.path.join(out_dir, 'metrics.json'), 'w'), indent=2)

        return avg
    return None


def main():
    parser = argparse.ArgumentParser(description='改进模型测试')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重路径 (best.pth)')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='测试图像目录，与--mask_dir配合使用')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='真值掩码目录（可选，有则输出指标）')
    parser.add_argument('--single_img', type=str, default=None,
                        help='单张图片路径')
    parser.add_argument('--output', type=str, default='improved/results',
                        help='结果输出目录')
    parser.add_argument('--size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--split', type=str, default=None,
                        choices=['easy', 'hard', 'all'],
                        help='快捷测试：easy / hard / all (基于预定义数据集)')
    parser.add_argument('--dataset', type=str, default='trans10k',
                        choices=['trans10k', 'data_test'],
                        help='使用--split时选择数据集')
    args = parser.parse_args()

    # device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # load the model
    print(f"加载模型: {args.model_path}")
    model = ImprovedGlassNet().to(device)
    state = torch.load(args.model_path, map_location=device)
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    print(f"加载权重: {len(filtered)}/{len(state)} 匹配")

    model.eval()
    os.makedirs(args.output, exist_ok=True)

    
    # quick test via --split
    
    if args.split:
        splits_to_test = ['easy', 'hard'] if args.split == 'all' else [args.split]
        all_results = {}

        for split_name in splits_to_test:
            avg = run_split_test(model, device, split_name, args.dataset, args)
            if avg:
                all_results[split_name] = avg

        # save the summary
        if all_results:
            summary_path = os.path.join(args.output, f'{args.dataset}_summary.json')
            json.dump(all_results, open(summary_path, 'w'), indent=2)
            print(f"\n汇总结果: {summary_path}")
        return

    
    # single-image test
    
    if args.single_img:
        prob = predict_single(model, args.single_img, device, args.size)
        name = os.path.splitext(os.path.basename(args.single_img))[0]
        Image.fromarray((prob * 255).astype(np.uint8)).save(os.path.join(args.output, f'{name}_prob.png'))
        Image.fromarray((prob > 0.5).astype(np.uint8) * 255).save(os.path.join(args.output, f'{name}_bin.png'))
        print(f"结果保存到 {args.output}/")
        return

    
    # custom-directory test
    
    if args.image_dir:
        dataset = TestDataset(args.image_dir, args.mask_dir, args.size)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
        metrics = evaluate(model, loader, device)
        if metrics:
            avg = {k: float(np.mean([m[k] for m in metrics])) for k in ['iou', 'f1', 'mae']}
            print(f"结果 ({len(metrics)} 样本): F1={avg['f1']:.4f}, IoU={avg['iou']:.4f}, MAE={avg['mae']:.4f}")
            json.dump({'model': args.model_path, 'num_samples': len(metrics), 'avg_metrics': avg},
                      open(os.path.join(args.output, 'test_results.json'), 'w'), indent=2)
        return

    print("请提供 --image_dir, --single_img, 或 --split")


if __name__ == '__main__':
    main()