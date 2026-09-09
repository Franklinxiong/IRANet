#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB-only glass detection training script.

Trains the RGB-only glass segmentation model, targeting low-light and
weakly reflective scenes. Two setups are supported:
1. Apple Silicon Mac (ARM, no CUDA).
2. Windows + NVIDIA GTX 3060 (CUDA 12.6).

Highlights:
- RGB-only input, no depth required.
- A composite loss function.
- Multi-scale training.
- Mixed-precision training (CUDA only).
- Environment-aware configuration.
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

# project imports
from data_loaderd import RescaleT, RandomCrop, RandomHorizontalFlip, ToTensorLab, SalObjDataset
from lr_scheduler import LR_Scheduler
from lovasz_losses import lovasz_hinge
import pytorch_ssim
import pytorch_iou
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

# the RGB-only model
from model_rgb_only import RGBOnlyGlassNet

# image utilities
import numpy as np
from skimage import io

# - environment detection and configuration
def detect_environment():
    """
    Detect the current environment and return the right configuration.

    Returns:
    dict: keys include:
    - 'is_mac_m4': bool, running on an Apple Silicon Mac.
    - 'is_windows_cuda': bool, Windows with CUDA.
    - 'device': torch.device.
    - 'use_amp': bool, mixed precision is enabled.
    - 'num_workers': int, workers for the data loader.
    """
    env_info = {
        'is_mac_m4': False,
        'is_windows_cuda': False,
        'device': torch.device('cpu'),
        'use_amp': False,
        'num_workers': 0
    }
    
    # detect platform and device
    system = platform.system()
    machine = platform.machine()
    
    print(f"系统: {system}, 架构: {machine}")
    
    if system == 'Darwin' and machine == 'arm64':
        # Apple Silicon Mac
        env_info['is_mac_m4'] = True
        env_info['device'] = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        env_info['num_workers'] = 0  # MPS does not allow multi-process loading
        print("检测到MacBook M4环境，使用MPS设备" if torch.backends.mps.is_available() else "检测到MacBook M4环境，使用CPU")
    
    elif system == 'Windows' and torch.cuda.is_available():
        # Windows + CUDA
        env_info['is_windows_cuda'] = True
        env_info['device'] = torch.device('cuda:0')
        env_info['use_amp'] = True  # enable AMP on CUDA
        env_info['num_workers'] = 16  # multi-process loading is fine on Windows
        
        # configure CUDA
        cudnn.enabled = True
        cudnn.benchmark = True
        torch.cuda.set_device(0)
        
        print(f"检测到Windows CUDA环境，使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    
    else:
        # fallback: Linux or unknown
        if torch.cuda.is_available():
            env_info['device'] = torch.device('cuda:0')
            env_info['use_amp'] = True
            env_info['num_workers'] = 16
            print(f"检测到CUDA环境，使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            env_info['device'] = torch.device('cpu')
            env_info['num_workers'] = 4
            print("检测到CPU环境")
    
    return env_info


# 1. loss functions

# base loss
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
bce_loss_with_logits = nn.BCEWithLogitsLoss(size_average=True)


def bce_iou_loss(pred, target):
    """
    Combined BCE and IoU loss.

    Arguments:
    pred (torch.Tensor): model output, [B, 1, H, W].
    target (torch.Tensor): ground truth, [B, 1, H, W].

    Returns:
    torch.Tensor: combined loss.
    """
    pred = torch.sigmoid(pred)
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out
    return loss


def structure_loss(pred, mask):
    """
    Structure-aware loss that focuses on boundary regions.

    Arguments:
    pred (torch.Tensor): model output.
    mask (torch.Tensor): ground-truth mask.

    Returns:
    torch.Tensor: structural loss.
    """
    # boundary pixels get a higher weight
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    
    # weighted BCE
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    
    # weighted IoU
    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    
    return (wbce + wiou).mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss that down-weights easy pixels.

    Arguments:
    pred (torch.Tensor): model output.
    target (torch.Tensor): ground truth.
    alpha (float): balances positive and negative samples.
    gamma (float): modulation factor; larger values focus more on hard pixels.

    Returns:
    torch.Tensor: focal loss.
    """
    pred_sigmoid = torch.sigmoid(pred)
    pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
    focal_weight = alpha * (1 - pt) ** gamma
    
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    focal_loss = focal_weight * bce
    
    return focal_loss.mean()


def edge_aware_loss(pred, target, edge_weight=3.0):
    """
    Edge-aware loss for sharper boundaries.

    Arguments:
    pred (torch.Tensor): model output.
    target (torch.Tensor): ground truth.
    edge_weight (float): weight of the edge term.

    Returns:
    torch.Tensor: edge-aware loss.
    """
    # edges of the target, from the gap between max and average pooling
    target_edges = F.max_pool2d(target, kernel_size=3, stride=1, padding=1) - \
                   F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    target_edges = torch.abs(target_edges)
    
    # edges of the prediction
    pred_edges = F.max_pool2d(torch.sigmoid(pred), kernel_size=3, stride=1, padding=1) - \
                 F.avg_pool2d(torch.sigmoid(pred), kernel_size=3, stride=1, padding=1)
    pred_edges = torch.abs(pred_edges)
    
    # L1 edge loss
    edge_loss = F.l1_loss(pred_edges, target_edges)
    
    return edge_loss * edge_weight


def enhanced_loss(pred, target):
    """
    Combined loss tailored to glass detection.

    Arguments:
    pred (torch.Tensor): model output.
    target (torch.Tensor): ground truth.

    Returns:
    torch.Tensor: total loss.
    """
    # base term (BCE + IoU)
    bce = bce_iou_loss(pred, target)
    
    # focal term to emphasise hard pixels
    focal = focal_loss(pred, target)
    
    # edge-aware term for sharper boundaries
    edge = edge_aware_loss(pred, target)
    
    # structural term for global consistency
    struct = structure_loss(pred, target)
    
    # weighted sum, weights picked empirically
    total_loss = bce + 0.3 * focal + 0.2 * edge + 0.2 * struct
    
    return total_loss


def muti_bce_loss_fusion(d1, d2, d3, d4, labels_v):
    """
    Multi-scale loss over the four decoder outputs.

    Arguments:
    d1, d2, d3, d4 (torch.Tensor): decoder stage outputs.
    labels_v (torch.Tensor): ground truth.

    Returns:
    tuple: (last-decoder loss, total loss).
    """
    loss1 = enhanced_loss(d1, labels_v)
    loss2 = enhanced_loss(d2, labels_v)
    loss3 = enhanced_loss(d3, labels_v)
    loss4 = enhanced_loss(d4, labels_v)
    
    loss = loss1 + loss2 + loss3 + loss4
    
    # per-stage losses for debugging
    print("多尺度损失 - l1: %3f, l2: %3f, l3: %3f, l4: %3f" % (
        loss1.item(), loss2.item(), loss3.item(), loss4.item()))
    
    return loss4, loss  # final-decoder loss and the total loss


# 2. dataset paths

# experiment tag used in saved model names
exp_name = 'rgb_only_model'

# adjust these paths to match your dataset layout
# expected data folder layout:
# data/
# ├── train/train/images/    # training images
# │   └── masks/             # training masks
# ├── validation/validation/easy/ and hard/   # validation
# └── test/test/easy/ and hard/       # test

data_dir = 'data/train/train/'  # training root
tra_image_dir = 'images/'        # training image subdir
tra_label_dir = 'masks/'         # training mask subdir

image_ext = '.jpg'               # image extension
label_ext = '.png'               # mask extension

# checkpoint output dir
model_dir = "./saved_models/{}/".format(exp_name)

# training settings
epoch_num = 200                  # number of epochs
batch_size_train = 14            # training batch size, tune to your GPU memory
batch_size_val = 1               # validation batch size
train_num = 0                    # number of training samples (computed)
val_num = 0                      # number of validation samples (computed)
lr = 1e-4                        # initial learning rate
train_size = 384                 # training image size

# list the training images
tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, '*' + image_ext))

# build the matching mask list
tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = os.path.basename(img_path)
    
    # filename without its extension
    name_without_ext = os.path.splitext(img_name)[0]
    
    # construct the mask path
    # masks are named like "1001_mask.png" while images are "1001.jpg"
    # so we take the numeric part of the image name and append "_mask.png"
    mask_path = os.path.join(data_dir, tra_label_dir, name_without_ext + "_mask" + label_ext)
    tra_lbl_name_list.append(mask_path)
    
    # drop images without a mask
    if not os.path.exists(mask_path):
        print(f"警告: 掩码文件不存在: {mask_path}")

# count the usable training samples
train_num = len(tra_img_name_list)
print(f"找到 {train_num} 个训练样本")


class RGBOnlyDataset(SalObjDataset):
    """
    RGB-only dataset, a SalObjDataset subclass with depth removed.

    Provides dummy depth and depth-missing values (zeros/ones) so the rest
    of the original data pipeline keeps working unchanged.
    """
    
    def __getitem__(self, idx):
        """
        Fetch one training sample.

        Arguments:
        idx (int): sample index.

        Returns:
        dict: image, label, depth and depth-missing mark.
        """
        # load the image
        if not os.path.isfile(self.image_name_list[idx]):
            raise FileNotFoundError(f"图像文件不存在: {self.image_name_list[idx]}")
        image = io.imread(self.image_name_list[idx])
        
        # load the label
        if 0 == len(self.label_name_list):
            label_3 = np.zeros(image.shape)
        else:
            if not os.path.isfile(self.label_name_list[idx]):
                raise FileNotFoundError(f"标签文件不存在: {self.label_name_list[idx]}")
            label_3 = io.imread(self.label_name_list[idx])
        
        # make sure the label is 2D
        label = np.zeros(label_3.shape[0:2])
        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]  # take the first channel
        elif 2 == len(label_3.shape):
            label = label_3
        
        # sanity check on image/label shapes
        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]
        
        # dummy depth and depth-missing maps
        # RGB-only training: no real depth is needed
        dep = np.zeros_like(label, dtype=np.float32)
        depm = np.ones_like(label, dtype=np.float32)  # depth_missing = 1 marks depth as unavailable
        
        sample = {
            'image': image,
            'label': label,
            'depth': dep,
            'depthmissing': depm
        }
        
        # apply the augmentation
        if self.transform:
            sample = self.transform(sample)
        
        return sample


# 7. learning-rate schedule

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=5):
    """
    Exponential-decay learning-rate schedule.

    Arguments:
    optimizer (torch.optim.Optimizer): the optimizer.
    init_lr (float): initial learning rate.
    epoch (int): current epoch.
    decay_rate (float): multiplicative decay, default 0.1.
    decay_epoch (int): decay interval, default 5 epochs.

    Notes:
    The learning rate is multiplied by decay_rate every decay_epoch epochs.
    """
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
    print(f"第{epoch}轮学习率调整为: {optimizer.param_groups[0]['lr']:.6f}")

# 9. end of training

def main():
    """
    Main training routine, guarded by __name__ == '__main__' so it never
    runs on import.
    """
    # 3. data loaders
    
    # augmentation pipeline
    transform = transforms.Compose([
        RescaleT(400),              # resize to 400px
        RandomCrop(384),            # random 384x384 crop
        RandomHorizontalFlip(0.5),  # flips with p=0.5
        ToTensorLab(flag=0)         # to tensor
    ])
    
    # build the dataset
    salobj_dataset = RGBOnlyDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        dep_name_list=[],  # no depth data needed
        transform=transform
    )
    
    # detect the environment and pick a config
    env_config = detect_environment()
    
    # build the data loaders
    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=env_config['num_workers']  # the number of workers depends on the platform
    )
    
    # 4. model
    
    print("=" * 50)
    print("初始化模型...")
    
    # create the model
    # 'resnext_101_32x4d.pth' is a pretrained ResNeXt101 checkpoint
    # if it is missing the model is initialised randomly
    net = RGBOnlyGlassNet('resnext_101_32x4d.pth')
    print("RGBOnlyGlassNet模型定义完成.")
    
    # move the model to the target device
    net.to(env_config['device'])
    print(f"模型已移动到设备: {env_config['device']}")
    
    # 5. optimizer
    
    print("=" * 50)
    print("定义优化器...")
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # 6. mixed-precision setup
    
    # enable AMP only on CUDA when requested
    if env_config['use_amp'] and env_config['device'].type == 'cuda':
        try:
            from apex import amp
            net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
            print("混合精度训练(AMP)已初始化 (O2优化级别)")
        except ImportError:
            print("警告: apex库未安装，混合精度训练不可用")
            env_config['use_amp'] = False
    else:
        env_config['use_amp'] = False
        print("混合精度训练未启用")
    
    # 8. training loop
    
    print("=" * 50)
    print("开始训练...")
    
    # running statistics
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    
    # scales used for multi-scale training
    size_rates = [0.75, 1, 1.25]  # multi-scale training: 0.75x, 1x, 1.25x
    
    # switch the model to training mode
    net.train()
    
    # training loop
    for epoch in range(1, epoch_num + 1):
        print(f"\n{'='*60}")
        print(f"开始第 {epoch}/{epoch_num} 轮训练")
        print(f"{'='*60}")
        
        for batch_idx, data in enumerate(salobj_dataloader):
            # multi-scale training loop
            for rate in size_rates:
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1
                
                # fetch a batch
                inputs, labels, depth, depth_missing = data['image'], data['label'], data['depth'], data['depthmissing']
                
                # to float tensor
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
                depth = depth.type(torch.FloatTensor)
                depth_missing = depth_missing.type(torch.FloatTensor)
                
                # move the batch to the device
                inputs_v = Variable(inputs.to(env_config['device']), requires_grad=False)
                labels_v = Variable(labels.to(env_config['device']), requires_grad=False)
                depth_v = Variable(depth.to(env_config['device']), requires_grad=False)
                depth_missing_v = Variable(depth_missing.to(env_config['device']), requires_grad=False)
                
                # multi-scale resize
                trainsize = int(round(train_size * rate / 32) * 32)
                if rate != 1:
                    # resize the input
                    inputs_v = F.interpolate(inputs_v, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    labels_v = F.interpolate(labels_v, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    depth_v = F.interpolate(depth_v, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    depth_missing_v = F.interpolate(depth_missing_v, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                
                # zero the gradients
                optimizer.zero_grad()
                
                # forward pass
                # RGBOnlyGlassNet returns only the 4 decoder outputs (no depth branch)
                d1, d2, d3, d4 = net(inputs_v, depth_v, depth_missing_v)
                
                # compute the loss
                loss2, loss = muti_bce_loss_fusion(d1, d2, d3, d4, labels_v)
                
                # backward
                if env_config['use_amp']:
                    # mixed-precision step
                    try:
                        from apex import amp
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    except ImportError:
                        # fall back to a plain backward pass if apex is missing
                        loss.backward()
                else:
                    # plain backward pass
                    loss.backward()
                
                # step the optimizer
                optimizer.step()
                
                # log the loss only at the standard scale
                if rate == 1:
                    running_loss += loss.item()
                    running_tar_loss += loss2.item()
                    
                    # log every 10 batches
                    if batch_idx % 10 == 0:
                        avg_loss = running_loss / ite_num4val
                        avg_tar_loss = running_tar_loss / ite_num4val
                        print(f"[epoch: {epoch:3d}/{epoch_num:3d}, "
                              f"batch: {(batch_idx+1)*batch_size_train:5d}/{train_num:5d}, "
                              f"ite: {ite_num:5d}] "
                              f"train loss: {avg_loss:.5f}, tar: {avg_tar_loss:.5f}, "
                              f"lr: {optimizer.param_groups[0]['lr']:.6f}")
        
        # adjust the learning rate at the end of each epoch
        adjust_lr(optimizer, lr, epoch, 0.9, int(epoch_num * 0.6))
        
        # checkpoint every 10 epochs
        if epoch % 10 == 0:
            # make sure the checkpoint dir exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                print(f"创建模型保存目录: {model_dir}")
            
            # save the checkpoint
            model_path = os.path.join(model_dir, f"rgb_only_epoch_{epoch}_loss_{running_loss/ite_num4val:.4f}.pth")
            torch.save(net.state_dict(), model_path)
            print(f"模型已保存到: {model_path}")
            
            # reset the running statistics
            running_loss = 0.0
            running_tar_loss = 0.0
            ite_num4val = 0
            
            # keep the model in training mode
            net.train()
        
        # lightweight validation every 5 epochs (optional)
        if epoch % 5 == 0:
            print(f"第{epoch}轮训练完成，建议进行验证集测试...")
    
    print('=' * 60)
    print('恭喜！RGB-only玻璃检测模型训练完成！')
    print('=' * 60)
    
    # save the final weights
    final_model_path = os.path.join(model_dir, "rgb_only_final_model.pth")
    torch.save(net.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # optionally save the model architecture summary
    model_info_path = os.path.join(model_dir, "model_info.txt")
    with open(model_info_path, 'w') as f:
        f.write(f"RGB-only玻璃检测模型训练信息\n")
        f.write(f"训练时间: {epoch_num}轮\n")
        f.write(f"批次大小: {batch_size_train}\n")
        f.write(f"学习率: {lr}\n")
        f.write(f"训练尺寸: {train_size}\n")
        f.write(f"环境: {env_config}\n")
        f.write(f"训练样本数: {train_num}\n")
    print(f"模型信息已保存到: {model_info_path}")
    
    print("\n训练脚本执行完毕！")
    print("下一步建议:")
    print("1. 使用batch_inference_test.py测试模型性能")
    print("2. 使用realtime_camera_inference.py进行实时摄像头测试")
    print("3. 使用evaluation_fast.py评估模型指标")


if __name__ == '__main__':
    main()
