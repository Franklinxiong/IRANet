"""
Improved training script v4.0 (CUDA 11.8 + RTX 4090).

Changes over the baseline:
1. Cosine annealing with a warmup.
2. Real validation, the best model is saved.
3. Stronger augmentation (ColorJitter and friends).
4. Multi-scale training.
5. EMA weight averaging.
6. Early stopping.
7. Mixed-precision (AMP) by default.
8. Gradient accumulation.
9. Warmup + CosineAnnealing + ReduceLROnPlateau in one schedule.
10. Resume from a checkpoint via --resume.
11. Automatic test on data_test (easy + hard) after training, reporting
F1/IoU/precision/recall.
12. Old checkpoints are pruned, only the best weights are kept.
"""

import os
import sys
import re
import gc
import glob
import json
import argparse
import numpy as np
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improved.model import ImprovedGlassNet
from pytorch_iou import IOU
from pytorch_ssim import SSIM



# preset test split paths

TEST_DATASETS = {
    'easy':   ('data/test/test/easy/images',   'data/test/test/easy/masks'),
    'hard':   ('data/test/test/hard/images',   'data/test/test/hard/masks'),
}



# dataset


class GlassTrainDataset(Dataset):
    """
    Training dataset with ColorJitter augmentation and multi-scale crops.
    """
    def __init__(self, image_dir, mask_dir, size=384, augment=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.augment = augment

        self.images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')) +
                             glob.glob(os.path.join(image_dir, '*.png')))
        print(f"找到 {len(self.images)} 个训练图像")

        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)

    def __len__(self):
        return len(self.images)

    def _load_mask(self, img_path):
        name = os.path.splitext(os.path.basename(img_path))[0]
        candidates = [
            os.path.join(self.mask_dir, f"{name}_mask.png"),
            os.path.join(self.mask_dir, f"{name}.png"),
            os.path.join(self.mask_dir, f"{name}_mask.jpg"),
            os.path.join(self.mask_dir, f"{name}.jpg"),
        ]
        for c in candidates:
            if os.path.exists(c):
                mask = Image.open(c).convert('L')
                return np.array(mask)
        return np.zeros((self.size, self.size), dtype=np.uint8)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        # multi-scale random resizing
        if self.augment and np.random.rand() > 0.25:
            scales = [384, 320, 288, 256, 416, 448]
            scale = scales[np.random.randint(len(scales))]
            image = image.resize((scale, scale), Image.BILINEAR)
        else:
            scale = self.size
            image = image.resize((scale, scale), Image.BILINEAR)

        image = np.array(image).astype(np.float32) / 255.0

        mask = self._load_mask(img_path)
        mask = np.array(Image.fromarray(mask).resize(
            (scale, scale), Image.NEAREST))
        mask = (mask > 30).astype(np.float32)

        if self.augment:
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
            img_pil = self.color_jitter(img_pil)
            image = np.array(img_pil).astype(np.float32) / 255.0

            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

            if np.random.rand() > 0.75:
                k = np.random.randint(1, 4)
                image = np.rot90(image, k, axes=(0, 1)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        if mask.size(2) != self.size:
            image = F.interpolate(image.unsqueeze(0), (self.size, self.size),
                                  mode='bilinear', align_corners=True).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0), (self.size, self.size),
                                 mode='nearest').squeeze(0)

        return {'image': image, 'label': mask, 'name': os.path.basename(img_path)}


class GlassEvalDataset(Dataset):
    """
    Validation and test dataset.
    """
    def __init__(self, image_dir, mask_dir, size=384):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')) +
                             glob.glob(os.path.join(image_dir, '*.png')))
        print(f"找到 {len(self.images)} 个评估图像")

    def __len__(self):
        return len(self.images)

    def _load_mask(self, img_path):
        name = os.path.splitext(os.path.basename(img_path))[0]
        candidates = [
            os.path.join(self.mask_dir, f"{name}_mask.png"),
            os.path.join(self.mask_dir, f"{name}.png"),
        ]
        for c in candidates:
            if os.path.exists(c):
                mask = Image.open(c).convert('L')
                return np.array(mask)
        return np.zeros((self.size, self.size), dtype=np.uint8)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.size, self.size), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0

        mask = self._load_mask(img_path)
        mask = np.array(Image.fromarray(mask).resize(
            (self.size, self.size), Image.NEAREST))
        mask = (mask > 30).astype(np.float32) if mask.max() > 1 else mask.astype(np.float32)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return {'image': image, 'label': mask, 'name': os.path.basename(img_path)}



# loss: BCE + IoU + SSIM + edge loss


class Criterion(nn.Module):
    def __init__(self, weight_iou=0.3, weight_ssim=0.1, weight_edge=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.iou = IOU()
        self.ssim = SSIM()
        self.weight_iou = weight_iou
        self.weight_ssim = weight_ssim
        self.weight_edge = weight_edge

    def _edge_loss(self, pred, target):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-7)
        edge_weight = target_edge.detach() + 1.0
        return F.binary_cross_entropy_with_logits(pred, target, weight=edge_weight).mean()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        prob = torch.sigmoid(pred)
        iou_loss = self.iou(prob, target)
        ssim_loss = self.ssim(prob, target)
        edge_loss = self._edge_loss(pred, target)
        return bce + self.weight_iou * iou_loss + self.weight_ssim * (1 - ssim_loss) + self.weight_edge * edge_loss



# EMA


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}



# evaluation (F1, IoU, precision, recall, MAE)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    criterion = nn.BCEWithLogitsLoss()

    all_f1, all_iou, all_precision, all_recall, all_mae = [], [], [], [], []
    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        outputs = model(images)
        pred = outputs[0]

        total_loss += criterion(pred, labels).item()
        prob = torch.sigmoid(pred).cpu().numpy()
        gt = labels.cpu().numpy()

        for i in range(prob.shape[0]):
            p = (prob[i, 0] > 0.5).flatten()
            g = (gt[i, 0] > 0.5).flatten()
            tp = (p & g).sum()
            fp = (p & ~g).sum()
            fn = (~p & g).sum()
            eps = 1e-7
            all_iou.append(float((tp / (tp + fp + fn + eps)).item()))
            all_precision.append(float((tp / (tp + fp + eps)).item()))
            all_recall.append(float((tp / (tp + fn + eps)).item()))
            all_f1.append(float((2 * tp / (2 * tp + fp + fn + eps)).item()))
            all_mae.append(float(np.abs(prob[i, 0] - gt[i, 0]).mean()))
        num_batches += 1

    n = len(all_f1)
    return {
        'loss': total_loss / max(num_batches, 1),
        'f1': np.mean(all_f1) if n > 0 else 0.0,
        'iou': np.mean(all_iou) if n > 0 else 0.0,
        'precision': np.mean(all_precision) if n > 0 else 0.0,
        'recall': np.mean(all_recall) if n > 0 else 0.0,
        'mae': np.mean(all_mae) if n > 0 else 0.0,
    }



# automatic testing after training


@torch.no_grad()
def test_on_dataset(model, device, split_name):
    img_dir, mask_dir = TEST_DATASETS[split_name]
    dataset = GlassEvalDataset(img_dir, mask_dir, size=384)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
    return evaluate(model, loader, device)


def auto_test(model, device, output_dir):
    print(f"\n{'='*60}")
    print(f"训练完成！自动测试 data_test 数据集...")
    print(f"{'='*60}")
    results = {}
    for split_name in ['easy', 'hard']:
        print(f"\n测试 {split_name}...")
        metrics = test_on_dataset(model, device, split_name)
        results[split_name] = {
            'f1': float(metrics['f1']),
            'iou': float(metrics['iou']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'mae': float(metrics['mae']),
        }
        print(f"  F1        = {metrics['f1']:.4f}")
        print(f"  IoU       = {metrics['iou']:.4f}")
        print(f"  Precision = {metrics['precision']:.4f}")
        print(f"  Recall    = {metrics['recall']:.4f}")
        print(f"  MAE       = {metrics['mae']:.4f}")

    result_path = os.path.join(output_dir, 'test_results.json')
    json.dump(results, open(result_path, 'w'), indent=2)
    print(f"\n测试结果已保存: {result_path}")
    return results



# train one epoch


def train_epoch(model, dataloader, optimizer, criterion, device, ema=None,
                scaler=None, use_amp=False, accumulation_steps=1):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = 0
            for i, out in enumerate(outputs):
                if out.size(2) != labels.size(2):
                    out = F.interpolate(out, size=labels.size()[2:],
                                        mode='bilinear', align_corners=True)
                weights_list = [1.0, 0.8, 0.6, 0.4]
                weight = weights_list[i] if i < len(weights_list) else 0.1
                loss += weight * criterion(out, labels)
            loss = loss / accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            optimizer.zero_grad()
            if ema:
                ema.update()

        total_loss += loss.item() * accumulation_steps
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch [{batch_idx+1}/{num_batches}] Loss: {loss.item() * accumulation_steps:.4f}")

    return total_loss / max(num_batches, 1)


def cleanup_checkpoints(output_dir, keep_best_only=True):
    if not keep_best_only:
        return
    patterns = ['latest.pth', 'checkpoint.pth', 'ckpt_ep*.pth']
    removed = []
    for pattern in patterns:
        for f in glob.glob(os.path.join(output_dir, pattern)):
            try:
                os.remove(f)
                removed.append(os.path.basename(f))
            except OSError:
                pass
    if removed:
        print(f"  清理旧checkpoint: {', '.join(removed)} (释放存储空间)")



# training curves (loss / F1 / IoU)


def plot_training_curves(log, output_dir, show=False):
    """
    Plot training loss / validation F1 / validation IoU and save as PNG.

    matplotlib is imported inside with the headless Agg backend, so plotting
    never breaks training even if matplotlib is missing or there is no
    display.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # headless backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib 未安装，跳过曲线可视化")
        return None

    if not log:
        print("⚠ 无训练日志，跳过曲线可视化")
        return None

    epochs = [e.get('epoch', i + 1) for i, e in enumerate(log)]
    train_loss = [e.get('train_loss') for e in log]
    val_f1 = [e.get('val_f1') for e in log]
    val_iou = [e.get('val_iou') for e in log]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) training loss
    axes[0].plot(epochs, train_loss, color='tab:red', marker='o',
                 markersize=3, linewidth=1.5, label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2) validation F1
    axes[1].plot(epochs, val_f1, color='tab:blue', marker='o',
                 markersize=3, linewidth=1.5, label='Val F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1')
    axes[1].set_title('Validation F1')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3) validation IoU
    axes[2].plot(epochs, val_iou, color='tab:green', marker='o',
                 markersize=3, linewidth=1.5, label='Val IoU')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].set_title('Validation IoU')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"训练曲线已保存: {plot_path}")
    return plot_path



# main


def main():
    parser = argparse.ArgumentParser(description='改进的RGB-only玻璃检测训练 v4.0')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批量大小 (24GB 4090+AMP推荐16，12GB推荐8)')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--size', type=int, default=384)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数 (16核CPU推荐8~10)')
    parser.add_argument('--backbone_path', type=str, default='resnext_101_32x4d.pth')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output_dir', type=str, default='improved/models')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='梯度累积步数')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='启用混合精度训练 (默认开启)')
    parser.add_argument('--no_amp', action='store_true',
                        help='禁用混合精度训练')
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint继续训练')
    parser.add_argument('--no_auto_test', action='store_true',
                        help='关闭训练结束后的自动测试')
    parser.add_argument('--keep_all_checkpoints', action='store_true',
                        help='保留所有checkpoint（默认只保留最佳权重）')
    args = parser.parse_args()

    # --no_amp overrides --amp
    if args.no_amp:
        args.amp = False

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
    print(f"PyTorch版本: {torch.__version__}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}, CUDA: {torch.version.cuda}")
    print(f"CPU核数检测: {os.cpu_count()}, 使用 num_workers={args.num_workers}")
    print(f"Batch Size: {args.batch_size}, AMP: {'开启' if args.amp else '关闭'}")

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, 'best.pth')
    latest_path = os.path.join(args.output_dir, 'latest.pth')

    # dataset
    train_dataset = GlassTrainDataset(
        'data/train/train/images', 'data/train/train/masks',
        size=args.size, augment=True)
    val_dataset = GlassEvalDataset(
        'data/validation/validation/easy/images',
        'data/validation/validation/easy/masks',
        size=args.size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # model
    backbone_path = args.backbone_path if os.path.exists(args.backbone_path) else None
    model = ImprovedGlassNet(backbone_path).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: 总{total_params/1e6:.2f}M, 可训练{trainable_params/1e6:.2f}M")

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    # LR schedule: warmup, then cosine annealing plus ReduceLROnPlateau
    
    warmup_epochs = 5
    cosine_epochs = args.epochs - warmup_epochs
    warmed_up = False

    def warmup_lambda(epoch):
        return min(1.0, (epoch + 1) / warmup_epochs)

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cosine_epochs, 1), eta_min=5e-7)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10,
        threshold=0.005, min_lr=5e-7)

    # loss
    criterion = Criterion(weight_iou=0.25, weight_ssim=0.1, weight_edge=0.05).to(device)

    # AMP
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("✓ 混合精度训练已启用 (AMP)")

    ema = EMA(model, decay=0.999)
    ema.register()

    
    # resume from a checkpoint
    
    start_epoch = 1
    best_f1 = 0
    patience = 30
    patience_counter = 0
    log = []

    if args.resume and os.path.exists(args.resume):
        print(f"\n{'='*60}")
        print(f"从 checkpoint 恢复训练: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            ema.shadow = ckpt['ema_shadow']
        else:
            print("  → 旧格式checkpoint，仅恢复模型权重")
            model.load_state_dict(ckpt)
            match = re.search(r'ckpt_ep(\d+)', args.resume)
            if match:
                start_epoch = int(match.group(1)) + 1

        if 'warmup_scheduler' in ckpt:
            warmup_scheduler.load_state_dict(ckpt['warmup_scheduler'])
            cosine_scheduler.load_state_dict(ckpt['cosine_scheduler'])
            plateau_scheduler.load_state_dict(ckpt['plateau_scheduler'])
        if 'warmed_up' in ckpt:
            warmed_up = ckpt['warmed_up']
        if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] is not None:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            best_f1 = ckpt.get('best_f1', 0)
            patience_counter = ckpt.get('patience_counter', 0)
            log = ckpt.get('log', [])

        print(f"  恢复: epoch {start_epoch}/{args.epochs}, 最佳F1={best_f1:.4f}")
        print(f"{'='*60}\n")
    elif args.resume:
        print(f"⚠ Checkpoint不存在: {args.resume}，从头开始")

    
    # training loop
    
    print(f"\n{'='*60}")
    print(f"开始训练 ({args.epochs} epochs)")
    if start_epoch > 1:
        print(f"从 epoch {start_epoch} 继续")
    print(f"Batch: {args.batch_size}, Warmup(5)→Cosine→ReduceLROnPlateau")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_idx = epoch - 1

        if epoch_idx == warmup_epochs and not warmed_up:
            warmed_up = True
            print(f"  → 切换到 CosineAnnealing (T_max={cosine_epochs})")

        print(f"\nEpoch {epoch}/{args.epochs}")
        lr_now = optimizer.param_groups[0]['lr']
        print(f"LR: {lr_now:.6f}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, ema,
            scaler=scaler, use_amp=use_amp,
            accumulation_steps=args.accumulation_steps)

        val_metrics = evaluate(model, val_loader, device)

        if warmed_up:
            cosine_scheduler.step()
            plateau_scheduler.step(val_metrics['f1'])
        else:
            warmup_scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val  F1={val_metrics['f1']:.4f} IoU={val_metrics['iou']:.4f} "
              f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} "
              f"MAE={val_metrics['mae']:.4f}")

        entry = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_f1': float(val_metrics['f1']),
            'val_iou': float(val_metrics['iou']),
            'val_precision': float(val_metrics['precision']),
            'val_recall': float(val_metrics['recall']),
            'val_mae': float(val_metrics['mae']),
            'lr': lr_now,
        }
        log.append(entry)

        # save a checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'warmup_scheduler': warmup_scheduler.state_dict(),
            'cosine_scheduler': cosine_scheduler.state_dict(),
            'plateau_scheduler': plateau_scheduler.state_dict(),
            'ema_shadow': ema.shadow,
            'scaler_state_dict': scaler.state_dict() if use_amp else None,
            'best_f1': best_f1,
            'patience_counter': patience_counter,
            'warmed_up': warmed_up,
            'log': log,
            'args': vars(args),
        }
        torch.save(ckpt, os.path.join(args.output_dir, 'checkpoint.pth'))
        torch.save(model.state_dict(), latest_path)

        # save the best model
        val_score = val_metrics['f1']
        if val_score > best_f1:
            best_f1 = val_score
            torch.save(model.state_dict(), best_path)
            ema.apply_shadow()
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_ema.pth'))
            ema.restore()
            print(f"  ✓ 新最佳! F1={val_score:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n提前停止 @ epoch {epoch}")
                break


    # save the log
    with open(os.path.join(args.output_dir, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    # plot the curves (loss / F1 / IoU)
    plot_training_curves(log, args.output_dir)

    print(f"\n{'='*60}")
    print(f"训练完成！最佳 F1: {best_f1:.4f}")
    print(f"最佳模型: {best_path}")
    print(f"{'='*60}")

    # prune old checkpoints
    if not args.keep_all_checkpoints:
        cleanup_checkpoints(args.output_dir)

    # automatic testing
    if not args.no_auto_test:
        best_ema_path = os.path.join(args.output_dir, 'best_ema.pth')
        if os.path.exists(best_ema_path):
            state = torch.load(best_ema_path, map_location=device)
            model_dict = model.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered)
            model.load_state_dict(model_dict)
            print(f"  ✓ 已加载最佳EMA权重")
        else:
            state = torch.load(best_path, map_location=device)
            model_dict = model.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered)
            model.load_state_dict(model_dict)

        auto_test(model, device, args.output_dir)

        if os.path.exists(best_ema_path):
            os.remove(best_ema_path)
            print(f"  已清理临时EMA权重")

    print(f"\n所有任务完成！")


if __name__ == '__main__':
    main()