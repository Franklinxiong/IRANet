"""
Training script for the ablation variants.

Each variant builds on the previous one, matching the ablation order:
  1. baseline          - no modules (backbone + cr + contrast + up)
  2. plus_illumination - add the illumination invariant module
  3. plus_reflection   - add reflection enhancement
  4. plus_attention    - add enhanced attention (full model)

Usage:
    python ablation/train_ablation.py --variant baseline
    python ablation/train_ablation.py --variant plus_illumination
    python ablation/train_ablation.py --variant plus_reflection
    python ablation/train_ablation.py --variant plus_attention
    python ablation/train_ablation.py --variant all
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import os
import glob
import sys
import platform
import numpy as np
from skimage import io
from tqdm import tqdm

# put the project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaderd import RescaleT, RandomCrop, RandomHorizontalFlip, ToTensorLab, SalObjDataset
import pytorch_ssim
import pytorch_iou
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from ablation.models import MODEL_VARIANTS


# - loss functions
bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def bce_iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    return bce_out + iou_out


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    pred_sigmoid = torch.sigmoid(pred)
    pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
    focal_weight = alpha * (1 - pt) ** gamma
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    return (focal_weight * bce).mean()


def edge_aware_loss(pred, target, edge_weight=3.0):
    target_edges = F.max_pool2d(target, kernel_size=3, stride=1, padding=1) - \
                   F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    target_edges = torch.abs(target_edges)
    pred_edges = F.max_pool2d(torch.sigmoid(pred), kernel_size=3, stride=1, padding=1) - \
                 F.avg_pool2d(torch.sigmoid(pred), kernel_size=3, stride=1, padding=1)
    pred_edges = torch.abs(pred_edges)
    return F.l1_loss(pred_edges, target_edges) * edge_weight


def enhanced_loss(pred, target):
    bce = bce_iou_loss(pred, target)
    focal = focal_loss(pred, target)
    edge = edge_aware_loss(pred, target)
    struct = structure_loss(pred, target)
    return bce + 0.3 * focal + 0.2 * edge + 0.2 * struct


def muti_bce_loss_fusion(d1, d2, d3, d4, labels_v):
    loss1 = enhanced_loss(d1, labels_v)
    loss2 = enhanced_loss(d2, labels_v)
    loss3 = enhanced_loss(d3, labels_v)
    loss4 = enhanced_loss(d4, labels_v)
    loss = loss1 + loss2 + loss3 + loss4
    return loss4, loss


# - learning rate scheduling
def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=5):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay


# - environment detection
def detect_environment():
    env_info = {
        'is_mac_m4': False,
        'is_windows_cuda': False,
        'device': torch.device('cpu'),
        'use_amp': False,
        'num_workers': 0
    }
    system = platform.system()
    machine = platform.machine()

    print(f"系统: {system}, 架构: {machine}")

    if system == 'Darwin' and machine == 'arm64':
        env_info['is_mac_m4'] = True
        env_info['device'] = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        env_info['num_workers'] = 0
        print("使用MPS设备" if torch.backends.mps.is_available() else "使用CPU")
    elif system == 'Windows' and torch.cuda.is_available():
        env_info['is_windows_cuda'] = True
        env_info['device'] = torch.device('cuda:0')
        env_info['use_amp'] = True
        env_info['num_workers'] = 8
        cudnn.enabled = True
        cudnn.benchmark = True
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        if torch.cuda.is_available():
            env_info['device'] = torch.device('cuda:0')
            env_info['use_amp'] = True
            env_info['num_workers'] = 8
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            env_info['device'] = torch.device('cpu')
            env_info['num_workers'] = 4
            print("使用CPU")
    return env_info


# - dataset classes
class RGBOnlyDataset(SalObjDataset):
    def __getitem__(self, idx):
        if not os.path.isfile(self.image_name_list[idx]):
            raise FileNotFoundError(f"图像文件不存在: {self.image_name_list[idx]}")
        image = io.imread(self.image_name_list[idx])

        if 0 == len(self.label_name_list):
            label_3 = np.zeros(image.shape)
        else:
            if not os.path.isfile(self.label_name_list[idx]):
                raise FileNotFoundError(f"标签文件不存在: {self.label_name_list[idx]}")
            label_3 = io.imread(self.label_name_list[idx])

        label = np.zeros(label_3.shape[0:2])
        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        dep = np.zeros_like(label, dtype=np.float32)
        depm = np.ones_like(label, dtype=np.float32)

        sample = {'image': image, 'label': label, 'depth': dep, 'depthmissing': depm}
        if self.transform:
            sample = self.transform(sample)
        return sample


# - train a single variant
def train_variant(args, env_config, variant_name, variant_display_name):
    """Train a single variant."""
    print(f"\n{'='*70}")
    print(f"训练变体: {variant_display_name}")
    print(f"{'='*70}")

    # checkpoint directory
    checkpoint_dir = os.path.join('ablation', 'checkpoints', variant_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # data loading
    tra_img_name_list = glob.glob(os.path.join(args.data_dir, args.tra_image_dir, '*' + args.image_ext))
    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(img_name)[0]
        mask_path = os.path.join(args.data_dir, args.tra_label_dir, name_without_ext + "_mask" + args.label_ext)
        tra_lbl_name_list.append(mask_path)
        if not os.path.exists(mask_path):
            print(f"警告: 掩码文件不存在: {mask_path}")

    train_num = len(tra_img_name_list)
    print(f"找到 {train_num} 个训练样本")
    if train_num == 0:
        print("错误: 没有找到训练样本")
        return None

    batch_size_train = args.batch_size

    # data augmentation
    transform = transforms.Compose([
        RescaleT(400),
        RandomCrop(args.train_size),
        RandomHorizontalFlip(0.5),
        ToTensorLab(flag=0)
    ])

    salobj_dataset = RGBOnlyDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        dep_name_list=[],
        transform=transform
    )

    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=env_config['num_workers']
    )

    # create the model
    print("\n初始化模型...")
    model_class = MODEL_VARIANTS[variant_name]
    net = model_class('resnext_101_32x4d.pth')
    net.to(env_config['device'])
    print(f"模型已移动到设备: {env_config['device']}")

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # mixed precision
    use_amp = env_config['use_amp'] and env_config['device'].type == 'cuda'
    if use_amp:
        try:
            from apex import amp
            net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
            print("混合精度训练(AMP)已初始化")
        except ImportError:
            print("警告: apex库未安装，混合精度训练不可用")
            use_amp = False
    else:
        print("混合精度训练未启用")

    # training loop
    print("\n开始训练...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    size_rates = [0.75, 1, 1.25]
    net.train()

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_tar_loss = 0.0
        num_batches = 0

        pbar = tqdm(salobj_dataloader, desc=f"Epoch {epoch:3d}/{args.epochs}", ncols=100, leave=False)
        for batch_idx, data in enumerate(pbar):
            for rate in size_rates:
                ite_num += 1
                ite_num4val += 1

                inputs, labels = data['image'], data['label']
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                inputs_v = Variable(inputs.to(env_config['device']), requires_grad=False)
                labels_v = Variable(labels.to(env_config['device']), requires_grad=False)

                trainsize = int(round(args.train_size * rate / 32) * 32)
                if rate != 1:
                    inputs_v = F.interpolate(inputs_v, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    labels_v = F.interpolate(labels_v, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                optimizer.zero_grad()
                d1, d2, d3, d4 = net(inputs_v)
                loss2, loss = muti_bce_loss_fusion(d1, d2, d3, d4, labels_v)

                if use_amp:
                    try:
                        from apex import amp
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    except ImportError:
                        loss.backward()
                else:
                    loss.backward()

                optimizer.step()

                if rate == 1:
                    running_loss += loss.item()
                    running_tar_loss += loss2.item()
                    epoch_loss += loss.item()
                    epoch_tar_loss += loss2.item()
                    num_batches += 1

            # update the progress bar
            avg_loss = running_loss / max(ite_num4val, 1)
            avg_tar = running_tar_loss / max(ite_num4val, 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", tar=f"{avg_tar:.4f}")

        # adjust the learning rate
        adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)

        # average loss for this epoch
        if num_batches > 0:
            epoch_avg_loss = epoch_loss / num_batches
            epoch_avg_tar = epoch_tar_loss / num_batches
        else:
            epoch_avg_loss = 0.0
            epoch_avg_tar = 0.0

        # save the best model
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            best_path = os.path.join(checkpoint_dir, f"{variant_name}_best.pth")
            torch.save(net.state_dict(), best_path)

        # checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{variant_name}_epoch_{epoch}_loss_{epoch_avg_loss:.4f}.pth")
            torch.save(net.state_dict(), checkpoint_path)

    # save the final model
    final_path = os.path.join(checkpoint_dir, f"{variant_name}_final.pth")
    torch.save(net.state_dict(), final_path)
    print(f"\n[{variant_display_name}] 训练完成! 最佳损失: {best_loss:.4f}")
    print(f"  最终模型: {final_path}")

    return final_path


# - main entry
def main():
    import argparse
    parser = argparse.ArgumentParser(description='消融实验训练脚本（从baseline逐个添加模块）')

    parser.add_argument('--variant', type=str, required=True,
                        choices=['baseline', 'plus_illumination', 'plus_reflection', 'plus_attention', 'all'],
                        help='要训练的模型变体；all表示依次训练所有变体')
    parser.add_argument('--data_dir', type=str, default='data/train/train/')
    parser.add_argument('--tra_image_dir', type=str, default='images/')
    parser.add_argument('--tra_label_dir', type=str, default='masks/')
    parser.add_argument('--image_ext', type=str, default='.jpg')
    parser.add_argument('--label_ext', type=str, default='.png')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_size', type=int, default=384)
    parser.add_argument('--decay_rate', type=float, default=0.9, help='学习率衰减率')
    parser.add_argument('--decay_epoch', type=int, default=120, help='学习率衰减周期（epoch）')

    args = parser.parse_args()

    # variant name mapping
    variant_display = {
        'baseline':          'Baseline（无三个模块）',
        'plus_illumination': 'Plus Illumination（加光照不变模块）',
        'plus_reflection':   'Plus Reflection（加反射增强模块）',
        'plus_attention':    'Plus Attention（加增强注意力模块，完整模型）',
    }

    print("=" * 70)
    print("消融实验训练（从baseline逐个添加模块）")
    print("=" * 70)
    print("\n流水线顺序: rgb_layer → illumination → reflection → attention → contrast")
    print("\n变体对照:")
    for v, d in variant_display.items():
        print(f"  {v:25s} - {d}")

    # environment detection
    env_config = detect_environment()

    # decide which variants to train
    if args.variant == 'all':
        variants_to_train = ['baseline', 'plus_illumination', 'plus_reflection', 'plus_attention']
    else:
        variants_to_train = [args.variant]

    # train each variant in order
    trained_models = {}
    for variant_name in variants_to_train:
        model_path = train_variant(args, env_config, variant_name, variant_display[variant_name])
        trained_models[variant_name] = model_path

    # print the summary
    print("\n" + "=" * 70)
    print("训练总结")
    print("=" * 70)
    for variant_name, model_path in trained_models.items():
        print(f"  {variant_name:25s}: {model_path if model_path else '训练失败'}")
    print("=" * 70)
    print("所有训练完成！")


if __name__ == '__main__':
    main()