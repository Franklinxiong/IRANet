"""
Ablation study for the RGB-only glass detection model.

Measures the contribution of the three specialised modules:
1. IlluminationInvariantModule
2. ReflectionEnhancementModule
3. EnhancedAttentionModule
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
from PIL import Image
import json
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import the reference model
from model_rgb_only import RGBOnlyGlassNet, SELayer



# ablation model variants


class RGBOnlyGlassNet_NoIllumination(RGBOnlyGlassNet):
    """
    Variant with the illumination-invariant module removed.
    """
    def __init__(self, backbone_path=None):
        super(RGBOnlyGlassNet_NoIllumination, self).__init__(backbone_path)
        
        # replace the illumination-invariant module with identity
        self.illumination_invariant_4 = nn.Identity()
        self.illumination_invariant_3 = nn.Identity()
        self.illumination_invariant_2 = nn.Identity()
        self.illumination_invariant_1 = nn.Identity()
    
    def forward(self, x):
        # 1. backbone features
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # 2. channel reduction
        rgb_layer4 = self.cr4(layer4)
        rgb_layer3 = self.cr3(layer3)
        rgb_layer2 = self.cr2(layer2)
        rgb_layer1 = self.cr1(layer1)

        # 3. illumination invariant (ablated out, identity instead)
        illum4 = self.illumination_invariant_4(rgb_layer4)
        illum3 = self.illumination_invariant_3(rgb_layer3)
        illum2 = self.illumination_invariant_2(rgb_layer2)
        illum1 = self.illumination_invariant_1(rgb_layer1)

        # 4. reflection enhancement
        reflect4 = self.reflection_enhancer_4(illum4)
        reflect3 = self.reflection_enhancer_3(illum3)
        reflect2 = self.reflection_enhancer_2(illum2)
        reflect1 = self.reflection_enhancer_1(illum1)

        # 5. attention
        attn4 = self.enhanced_attention_4(reflect4)
        attn3 = self.enhanced_attention_3(reflect3)
        attn2 = self.enhanced_attention_2(reflect2)
        attn1 = self.enhanced_attention_1(reflect1)

        # 6. multi-scale contrast features
        contrast4 = self.rgb_contrast_4(attn4)
        contrast3 = self.rgb_contrast_3(attn3)
        contrast2 = self.rgb_contrast_2(attn2)
        contrast1 = self.rgb_contrast_1(attn1)

        # 7. multi-scale decoder
        up4 = self.up_4(contrast4)
        layer4_predict = self.layer4_predict(up4)

        up3 = self.up_3(contrast3 + up4)
        layer3_predict = self.layer3_predict(up3)

        up2 = self.up_2(contrast2 + up3)
        layer2_predict = self.layer2_predict(up2)

        up1 = self.up_1(contrast1 + up2)
        layer1_predict = self.layer1_predict(up1)

        # 8. upsample to original size
        layer4_predict = F.interpolate(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.interpolate(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.interpolate(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.interpolate(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        return layer1_predict, layer2_predict, layer3_predict, layer4_predict


class RGBOnlyGlassNet_NoReflection(RGBOnlyGlassNet):
    """
    Variant with the reflection enhancement module removed.
    """
    def __init__(self, backbone_path=None):
        super(RGBOnlyGlassNet_NoReflection, self).__init__(backbone_path)
        
        # replace the reflection module with identity
        self.reflection_enhancer_4 = nn.Identity()
        self.reflection_enhancer_3 = nn.Identity()
        self.reflection_enhancer_2 = nn.Identity()
        self.reflection_enhancer_1 = nn.Identity()
    
    def forward(self, x):
        # 1. backbone features
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # 2. channel reduction
        rgb_layer4 = self.cr4(layer4)
        rgb_layer3 = self.cr3(layer3)
        rgb_layer2 = self.cr2(layer2)
        rgb_layer1 = self.cr1(layer1)

        # 3. illumination invariant
        illum4 = self.illumination_invariant_4(rgb_layer4)
        illum3 = self.illumination_invariant_3(rgb_layer3)
        illum2 = self.illumination_invariant_2(rgb_layer2)
        illum1 = self.illumination_invariant_1(rgb_layer1)

        # 4. reflection enhancement (ablated out, identity instead)
        reflect4 = self.reflection_enhancer_4(illum4)
        reflect3 = self.reflection_enhancer_3(illum3)
        reflect2 = self.reflection_enhancer_2(illum2)
        reflect1 = self.reflection_enhancer_1(illum1)

        # 5. attention
        attn4 = self.enhanced_attention_4(reflect4)
        attn3 = self.enhanced_attention_3(reflect3)
        attn2 = self.enhanced_attention_2(reflect2)
        attn1 = self.enhanced_attention_1(reflect1)

        # 6. multi-scale contrast features
        contrast4 = self.rgb_contrast_4(attn4)
        contrast3 = self.rgb_contrast_3(attn3)
        contrast2 = self.rgb_contrast_2(attn2)
        contrast1 = self.rgb_contrast_1(attn1)

        # 7. multi-scale decoder
        up4 = self.up_4(contrast4)
        layer4_predict = self.layer4_predict(up4)

        up3 = self.up_3(contrast3 + up4)
        layer3_predict = self.layer3_predict(up3)

        up2 = self.up_2(contrast2 + up3)
        layer2_predict = self.layer2_predict(up2)

        up1 = self.up_1(contrast1 + up2)
        layer1_predict = self.layer1_predict(up1)

        # 8. upsample to original size
        layer4_predict = F.interpolate(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.interpolate(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.interpolate(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.interpolate(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        return layer1_predict, layer2_predict, layer3_predict, layer4_predict


class RGBOnlyGlassNet_NoAttention(RGBOnlyGlassNet):
    """
    Variant with the enhanced attention module removed.
    """
    def __init__(self, backbone_path=None):
        super(RGBOnlyGlassNet_NoAttention, self).__init__(backbone_path)
        
        # replace the attention module with identity
        self.enhanced_attention_4 = nn.Identity()
        self.enhanced_attention_3 = nn.Identity()
        self.enhanced_attention_2 = nn.Identity()
        self.enhanced_attention_1 = nn.Identity()
    
    def forward(self, x):
        # 1. backbone features
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # 2. channel reduction
        rgb_layer4 = self.cr4(layer4)
        rgb_layer3 = self.cr3(layer3)
        rgb_layer2 = self.cr2(layer2)
        rgb_layer1 = self.cr1(layer1)

        # 3. illumination invariant
        illum4 = self.illumination_invariant_4(rgb_layer4)
        illum3 = self.illumination_invariant_3(rgb_layer3)
        illum2 = self.illumination_invariant_2(rgb_layer2)
        illum1 = self.illumination_invariant_1(rgb_layer1)

        # 4. reflection enhancement
        reflect4 = self.reflection_enhancer_4(illum4)
        reflect3 = self.reflection_enhancer_3(illum3)
        reflect2 = self.reflection_enhancer_2(illum2)
        reflect1 = self.reflection_enhancer_1(illum1)

        # 5. attention (ablated out, identity instead)
        attn4 = self.enhanced_attention_4(reflect4)
        attn3 = self.enhanced_attention_3(reflect3)
        attn2 = self.enhanced_attention_2(reflect2)
        attn1 = self.enhanced_attention_1(reflect1)

        # 6. multi-scale contrast features
        contrast4 = self.rgb_contrast_4(attn4)
        contrast3 = self.rgb_contrast_3(attn3)
        contrast2 = self.rgb_contrast_2(attn2)
        contrast1 = self.rgb_contrast_1(attn1)

        # 7. multi-scale decoder
        up4 = self.up_4(contrast4)
        layer4_predict = self.layer4_predict(up4)

        up3 = self.up_3(contrast3 + up4)
        layer3_predict = self.layer3_predict(up3)

        up2 = self.up_2(contrast2 + up3)
        layer2_predict = self.layer2_predict(up2)

        up1 = self.up_1(contrast1 + up2)
        layer1_predict = self.layer1_predict(up1)

        # 8. upsample to original size
        layer4_predict = F.interpolate(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.interpolate(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.interpolate(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.interpolate(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        return layer1_predict, layer2_predict, layer3_predict, layer4_predict


class RGBOnlyGlassNet_NoAll(RGBOnlyGlassNet):
    """
    Baseline variant with all three specialised modules removed.
    """
    def __init__(self, backbone_path=None):
        super(RGBOnlyGlassNet_NoAll, self).__init__(backbone_path)
        
        # baseline: all three modules replaced by identity
        self.illumination_invariant_4 = nn.Identity()
        self.illumination_invariant_3 = nn.Identity()
        self.illumination_invariant_2 = nn.Identity()
        self.illumination_invariant_1 = nn.Identity()
        
        self.reflection_enhancer_4 = nn.Identity()
        self.reflection_enhancer_3 = nn.Identity()
        self.reflection_enhancer_2 = nn.Identity()
        self.reflection_enhancer_1 = nn.Identity()
        
        self.enhanced_attention_4 = nn.Identity()
        self.enhanced_attention_3 = nn.Identity()
        self.enhanced_attention_2 = nn.Identity()
        self.enhanced_attention_1 = nn.Identity()
    
    def forward(self, x):
        # 1. backbone features
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # 2. channel reduction
        rgb_layer4 = self.cr4(layer4)
        rgb_layer3 = self.cr3(layer3)
        rgb_layer2 = self.cr2(layer2)
        rgb_layer1 = self.cr1(layer1)

        # 3. illumination invariant (ablated out)
        illum4 = self.illumination_invariant_4(rgb_layer4)
        illum3 = self.illumination_invariant_3(rgb_layer3)
        illum2 = self.illumination_invariant_2(rgb_layer2)
        illum1 = self.illumination_invariant_1(rgb_layer1)

        # 4. reflection enhancement (ablated out)
        reflect4 = self.reflection_enhancer_4(illum4)
        reflect3 = self.reflection_enhancer_3(illum3)
        reflect2 = self.reflection_enhancer_2(illum2)
        reflect1 = self.reflection_enhancer_1(illum1)

        # 5. attention (ablated out)
        attn4 = self.enhanced_attention_4(reflect4)
        attn3 = self.enhanced_attention_3(reflect3)
        attn2 = self.enhanced_attention_2(reflect2)
        attn1 = self.enhanced_attention_1(reflect1)

        # 6. multi-scale contrast features
        contrast4 = self.rgb_contrast_4(attn4)
        contrast3 = self.rgb_contrast_3(attn3)
        contrast2 = self.rgb_contrast_2(attn2)
        contrast1 = self.rgb_contrast_1(attn1)

        # 7. multi-scale decoder
        up4 = self.up_4(contrast4)
        layer4_predict = self.layer4_predict(up4)

        up3 = self.up_3(contrast3 + up4)
        layer3_predict = self.layer3_predict(up3)

        up2 = self.up_2(contrast2 + up3)
        layer2_predict = self.layer2_predict(up2)

        up1 = self.up_1(contrast1 + up2)
        layer1_predict = self.layer1_predict(up1)

        # 8. upsample to original size
        layer4_predict = F.interpolate(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.interpolate(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.interpolate(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.interpolate(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        return layer1_predict, layer2_predict, layer3_predict, layer4_predict



# datasets and loading


class GlassTestDataset(Dataset):
    """
    Glass detection test dataset.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # list all images
        self.image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')) + 
                                 glob.glob(os.path.join(image_dir, '*.png')))
        
        print(f"找到 {len(self.image_files)} 个测试图像")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # load the image
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(img_name)[0]
        
        # load and square the image
        image = Image.open(img_path).convert('RGB')
        # fixed 512x512 so the model sees a consistent input size
        image = image.resize((512, 512), Image.BILINEAR)
        image = np.array(image)
        
        # load the mask, trying several naming conventions
        mask_path = None
        mask_candidates = [
            os.path.join(self.mask_dir, f"{name_without_ext}_mask.png"),
            os.path.join(self.mask_dir, f"{name_without_ext}.png"),
            os.path.join(self.mask_dir, f"{name_without_ext}_mask.jpg"),
            os.path.join(self.mask_dir, f"{name_without_ext}.jpg"),
        ]
        
        for candidate in mask_candidates:
            if os.path.exists(candidate):
                mask_path = candidate
                break
        
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            # resize the mask to match
            mask = mask.resize((512, 512), Image.NEAREST)
            mask = np.array(mask)
        else:
            # fall back to an all-zero mask
            print(f"警告: 图像 {img_name} 没有对应的掩码文件")
            mask = np.zeros((512, 512), dtype=np.uint8)
        
        # these masks max out at 76, not 255, so the threshold is lower
        # binarise with a much lower threshold
        if mask.max() > 1:
            mask = (mask > 30).astype(np.float32)  # lower threshold
        else:
            mask = mask.astype(np.float32)
        
        # apply transforms
        if self.transform:
            transformed = self.transform({'image': image, 'label': mask})
            image = transformed['image']
            mask = transformed['label']
        else:
            # base transform: normalise and convert to tensor
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {
            'image': image,
            'label': mask,
            'name': img_name
        }


class ToTensorLab(object):
    """
    Converts samples to tensors.
    """
    def __init__(self, flag=0):
        self.flag = flag
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # enforce float32
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # enforce float32 too
        if label.dtype != np.float32:
            label = label.astype(np.float32)
        
        # ImageNet normalisation
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
        image = image / np.max(image)
        
        if image.shape[2] == 1:
            # grayscale
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            # colour
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        
        # labels: make sure they are 2D
        if np.max(label) < 1e-6:
            tmpLbl = label
        else:
            tmpLbl = label / np.max(label)
        
        # (H, W, C) -> (C, H, W)
        tmpImg = tmpImg.transpose((2, 0, 1))
        
        # single-channel labels need an extra axis
        if len(tmpLbl.shape) == 2:
            tmpLbl = tmpLbl[np.newaxis, :, :]  # add the channel axis
        
        # to tensor
        return {
            'image': torch.from_numpy(tmpImg).float(),
            'label': torch.from_numpy(tmpLbl).float()
        }



# evaluation


def compute_metrics(predictions, targets, threshold=0.5):
    """
    Compute evaluation metrics.

    Arguments:
    predictions: predicted probability maps [B, 1, H, W].
    targets: ground truth [B, 1, H, W].
    threshold: binarisation threshold.

    Returns:
    metrics: dictionary of metrics.
    """
    # must be numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # flatten
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # binarise
    pred_binary = (pred_flat > threshold).astype(np.int32)
    target_binary = (target_flat > 0.5).astype(np.int32)
    
    # score
    metrics = {}
    
    # basic stats
    eps = 1e-7
    tp = np.sum((pred_binary == 1) & (target_binary == 1))
    tn = np.sum((pred_binary == 0) & (target_binary == 0))
    fp = np.sum((pred_binary == 1) & (target_binary == 0))
    fn = np.sum((pred_binary == 0) & (target_binary == 1))
    
    # precision, recall, F1
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    # IoU (Jaccard index)
    iou = tp / (tp + fp + fn + eps)
    
    # accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    # Dice coefficient
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    
    # MAE
    mae = np.mean(np.abs(pred_flat - target_flat))
    
    metrics['precision'] = float(precision)
    metrics['recall'] = float(recall)
    metrics['f1_score'] = float(f1)
    metrics['iou'] = float(iou)
    metrics['accuracy'] = float(accuracy)
    metrics['dice'] = float(dice)
    metrics['mae'] = float(mae)
    metrics['tp'] = int(tp)
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    
    return metrics


def evaluate_model(model, dataloader, device, model_name="模型"):
    """
    Evaluate a model on a dataloader.

    Arguments:
    model: the model.
    dataloader: the data loader.
    device: device.
    model_name: name of the model.

    Returns:
    metrics: averaged metrics.
    all_metrics: per-sample metrics.
    """
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            names = batch['name']
            
            # forward pass
            outputs = model(images)
            
            # the finest decoder output (layer1_predict)
            predictions = torch.sigmoid(outputs[0])  # layer1_predict
            
            # per-sample metrics
            for i in range(predictions.shape[0]):
                pred = predictions[i:i+1]
                label = labels[i:i+1]
                
                metrics = compute_metrics(pred, label)
                metrics['name'] = names[i]
                all_metrics.append(metrics)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  {model_name}: 已处理 {batch_idx + 1}/{len(dataloader)} 批次")
    
    # average the metrics
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            if key != 'name' and key not in ['tp', 'tn', 'fp', 'fn']:
                values = [m[key] for m in all_metrics]
                avg_metrics[key] = float(np.mean(values))
        
        # summary stats
        total_tp = sum(m['tp'] for m in all_metrics)
        total_tn = sum(m['tn'] for m in all_metrics)
        total_fp = sum(m['fp'] for m in all_metrics)
        total_fn = sum(m['fn'] for m in all_metrics)
        
        avg_metrics['total_tp'] = total_tp
        avg_metrics['total_tn'] = total_tn
        avg_metrics['total_fp'] = total_fp
        avg_metrics['total_fn'] = total_fn
    
    return avg_metrics, all_metrics



# main


def main():
    parser = argparse.ArgumentParser(description='RGB-only玻璃检测模型消融实验')
    parser.add_argument('--test_easy_dir', type=str, default='data/test/test/easy',
                       help='easy测试集目录（包含images和masks子目录）')
    parser.add_argument('--test_hard_dir', type=str, default='data/test/test/hard',
                       help='hard测试集目录（包含images和masks子目录）')
    parser.add_argument('--model_path', type=str, default='optimized_models/best_rgb_only_model.pth',
                       help='预训练模型权重路径')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作进程数')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='结果输出目录')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu', 'mps'],
                       help='计算设备')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("RGB-only玻璃检测模型消融实验")
    print("=" * 70)
    
    # make the output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # pick the device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"使用CUDA设备: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("使用MPS设备 (Mac)")
        else:
            device = torch.device('cpu')
            print("使用CPU设备")
    else:
        device = torch.device(args.device)
        print(f"使用指定设备: {args.device}")
    
    # check the test directory
    test_dirs = [
        ('easy', args.test_easy_dir),
        ('hard', args.test_hard_dir)
    ]
    
    valid_test_dirs = []
    for name, test_dir in test_dirs:
        image_dir = os.path.join(test_dir, 'images')
        mask_dir = os.path.join(test_dir, 'masks')
        
        if os.path.exists(image_dir) and os.path.exists(mask_dir):
            valid_test_dirs.append((name, test_dir))
            print(f"找到测试集: {name} ({test_dir})")
        else:
            print(f"警告: 测试集目录不完整: {test_dir}")
            print(f"  图像目录存在: {os.path.exists(image_dir)}")
            print(f"  掩码目录存在: {os.path.exists(mask_dir)}")
    
    if not valid_test_dirs:
        print("错误: 没有找到有效的测试集目录")
        return
    
    # check the weights exist
    if not os.path.exists(args.model_path):
        print(f"错误: 模型权重文件不存在: {args.model_path}")
        return
    
    print(f"使用模型权重: {args.model_path}")
    
    # transforms
    transform = transforms.Compose([
        ToTensorLab(flag=0)
    ])
    
    # the variants to evaluate
    model_variants = [
        ('完整模型', RGBOnlyGlassNet),
        ('无光照不变模块', RGBOnlyGlassNet_NoIllumination),
        ('无反射增强模块', RGBOnlyGlassNet_NoReflection),
        ('无增强注意力模块', RGBOnlyGlassNet_NoAttention),
        ('无所有三个模块', RGBOnlyGlassNet_NoAll)
    ]
    
    # collect everything here
    all_results = {}
    
    # evaluate on each test set
    for test_name, test_dir in valid_test_dirs:
        print(f"\n{'='*70}")
        print(f"在 {test_name} 测试集上评估")
        print(f"{'='*70}")
        
        image_dir = os.path.join(test_dir, 'images')
        mask_dir = os.path.join(test_dir, 'masks')
        
        # datasets and loaders
        dataset = GlassTestDataset(image_dir, mask_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=args.num_workers)
        
        print(f"测试集大小: {len(dataset)} 个样本")
        
        test_results = {}
        
        # evaluate every variant
        for variant_name, model_class in model_variants:
            print(f"\n评估 {variant_name}...")
            
            try:
                # create the model
                model = model_class(backbone_path=None)
                
                # load the weights
                checkpoint = torch.load(args.model_path, map_location=device)
                
                # strict=False tolerates the expected key differences
                try:
                    model.load_state_dict(checkpoint, strict=False)
                    print(f"  权重加载成功 (strict=False)")
                except Exception as e:
                    print(f"  权重加载失败: {e}")
                    print(f"  使用随机初始化的模型")
                
                # move to device
                model = model.to(device)
                
                # evaluate
                avg_metrics, all_metrics = evaluate_model(model, dataloader, device, variant_name)
                
                # store the scores
                test_results[variant_name] = {
                    'avg_metrics': avg_metrics,
                    'num_samples': len(all_metrics)
                }
                
                # print the scores
                print(f"  {variant_name} 结果:")
                print(f"    样本数: {len(all_metrics)}")
                print(f"    F1分数: {avg_metrics.get('f1_score', 0):.4f}")
                print(f"    IoU: {avg_metrics.get('iou', 0):.4f}")
                print(f"    精确率: {avg_metrics.get('precision', 0):.4f}")
                print(f"    召回率: {avg_metrics.get('recall', 0):.4f}")
                print(f"    Dice系数: {avg_metrics.get('dice', 0):.4f}")
                print(f"    MAE: {avg_metrics.get('mae', 0):.4f}")
                
            except Exception as e:
                print(f"  评估 {variant_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        all_results[test_name] = test_results
    
    
    # report and plots
    
    
    print(f"\n{'='*70}")
    print("生成消融实验报告")
    print(f"{'='*70}")
    
    # text report
    report_path = os.path.join(args.output_dir, 'ablation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RGB-only玻璃检测模型消融实验报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型权重: {args.model_path}\n")
        f.write(f"设备: {device}\n")
        f.write(f"批次大小: {args.batch_size}\n\n")
        
        for test_name, test_results in all_results.items():
            f.write(f"{'='*70}\n")
            f.write(f"测试集: {test_name}\n")
            f.write(f"{'='*70}\n\n")
            
            # results table
            f.write(f"{'模型变体':<25} {'F1分数':<10} {'IoU':<10} {'精确率':<10} {'召回率':<10} {'Dice':<10} {'MAE':<10}\n")
            f.write("-" * 95 + "\n")
            
            for variant_name, result in test_results.items():
                metrics = result['avg_metrics']
                f.write(f"{variant_name:<25} ")
                f.write(f"{metrics.get('f1_score', 0):<10.4f} ")
                f.write(f"{metrics.get('iou', 0):<10.4f} ")
                f.write(f"{metrics.get('precision', 0):<10.4f} ")
                f.write(f"{metrics.get('recall', 0):<10.4f} ")
                f.write(f"{metrics.get('dice', 0):<10.4f} ")
                f.write(f"{metrics.get('mae', 0):<10.4f}\n")
            
            f.write("\n")
            
            # relative drop per variant
            if '完整模型' in test_results and len(test_results) > 1:
                f.write("相对性能下降（与完整模型相比）:\n")
                full_model_metrics = test_results['完整模型']['avg_metrics']
                full_f1 = full_model_metrics.get('f1_score', 0)
                full_iou = full_model_metrics.get('iou', 0)
                
                for variant_name, result in test_results.items():
                    if variant_name != '完整模型':
                        metrics = result['avg_metrics']
                        variant_f1 = metrics.get('f1_score', 0)
                        variant_iou = metrics.get('iou', 0)
                        
                        if full_f1 > 0:
                            f1_drop = (full_f1 - variant_f1) / full_f1 * 100
                        else:
                            f1_drop = 0
                        
                        if full_iou > 0:
                            iou_drop = (full_iou - variant_iou) / full_iou * 100
                        else:
                            iou_drop = 0
                        
                        f.write(f"  {variant_name}: F1下降 {f1_drop:.1f}%, IoU下降 {iou_drop:.1f}%\n")
                
                f.write("\n")
        
        # discussion
        f.write(f"{'='*70}\n")
        f.write("总结分析\n")
        f.write(f"{'='*70}\n\n")
        
        # how much each module matters
        for test_name, test_results in all_results.items():
            if '完整模型' in test_results and '无所有三个模块' in test_results:
                full_metrics = test_results['完整模型']['avg_metrics']
                baseline_metrics = test_results['无所有三个模块']['avg_metrics']
                
                full_f1 = full_metrics.get('f1_score', 0)
                baseline_f1 = baseline_metrics.get('f1_score', 0)
                
                if baseline_f1 > 0:
                    overall_improvement = (full_f1 - baseline_f1) / baseline_f1 * 100
                else:
                    overall_improvement = 0
                
                f.write(f"{test_name}测试集:\n")
                f.write(f"  三个模块整体提升: {overall_improvement:.1f}% (F1从{baseline_f1:.4f}提升到{full_f1:.4f})\n")
        
        f.write("\n")
        
        # rank the modules
        f.write("模块重要性分析（基于F1分数下降）:\n")
        for test_name, test_results in all_results.items():
            if '完整模型' in test_results:
                full_f1 = test_results['完整模型']['avg_metrics'].get('f1_score', 0)
                
                # drop per variant
                drops = {}
                for variant_name, result in test_results.items():
                    if variant_name != '完整模型':
                        variant_f1 = result['avg_metrics'].get('f1_score', 0)
                        if full_f1 > 0:
                            drop = (full_f1 - variant_f1) / full_f1 * 100
                        else:
                            drop = 0
                        drops[variant_name] = drop
                
                # sort by how much each module contributes
                sorted_drops = sorted(drops.items(), key=lambda x: x[1], reverse=True)
                
                f.write(f"  {test_name}: ")
                for i, (variant_name, drop) in enumerate(sorted_drops):
                    f.write(f"{variant_name.replace('无', '')}({drop:.1f}%)")
                    if i < len(sorted_drops) - 1:
                        f.write(" > ")
                f.write("\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("实验完成\n")
        f.write("=" * 70 + "\n")
    
    print(f"文本报告已保存: {report_path}")
    
    # visualise
    print("生成可视化图表...")
    
    for test_name, test_results in all_results.items():
        # bar chart
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{test_name}测试集 - 消融实验结果', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['f1_score', 'iou', 'precision', 'recall', 'dice', 'mae']
        metric_names = ['F1分数', 'IoU', '精确率', '召回率', 'Dice系数', 'MAE']
        
        variant_names = []
        metric_values = {metric: [] for metric in metrics_to_plot}
        
        for variant_name in ['完整模型', '无光照不变模块', '无反射增强模块', '无增强注意力模块', '无所有三个模块']:
            if variant_name in test_results:
                variant_names.append(variant_name)
                metrics = test_results[variant_name]['avg_metrics']
                for metric in metrics_to_plot:
                    metric_values[metric].append(metrics.get(metric, 0))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B8F71']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(variant_names, metric_values[metric], color=colors[:len(variant_names)])
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_ylabel('值' if metric != 'mae' else '误差')
            ax.tick_params(axis='x', rotation=45)
            
            # value labels on the bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            # y-axis limits
            if metric == 'mae':
                ax.set_ylim(0, max(metric_values[metric]) * 1.2)
            else:
                ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, f'{test_name}_ablation_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  图表已保存: {plot_path}")
        
        # radar chart
        if len(variant_names) > 1:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='polar')
            
            # prep the data
            angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot) - 1, endpoint=False).tolist()
            angles += angles[:1]  # close the loop
            
            for i, variant_name in enumerate(variant_names):
                values = []
                for metric in metrics_to_plot[:-1]:  # MAE excluded, its direction differs
                    values.append(metric_values[metric][i])
                values += values[:1]  # close the loop
                
                ax.plot(angles, values, 'o-', linewidth=2, label=variant_name, color=colors[i])
                ax.fill(angles, values, alpha=0.1, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_names[:-1])
            ax.set_ylim(0, 1.0)
            ax.set_title(f'{test_name}测试集 - 模块性能雷达图', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            radar_path = os.path.join(args.output_dir, f'{test_name}_radar_chart.png')
            plt.savefig(radar_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  雷达图已保存: {radar_path}")
    
    # dump detailed results as JSON
    json_path = os.path.join(args.output_dir, 'detailed_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        # make non-serialisable values JSON-safe
        json_results = {}
        for test_name, test_results in all_results.items():
            json_results[test_name] = {}
            for variant_name, result in test_results.items():
                json_results[test_name][variant_name] = {
                    'avg_metrics': result['avg_metrics'],
                    'num_samples': result['num_samples']
                }
        
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"详细结果已保存: {json_path}")
    
    print(f"\n{'='*70}")
    print("消融实验完成!")
    print(f"{'='*70}")
    
    # final summary
    print("\n最终总结:")
    for test_name, test_results in all_results.items():
        if '完整模型' in test_results and '无所有三个模块' in test_results:
            full_f1 = test_results['完整模型']['avg_metrics'].get('f1_score', 0)
            baseline_f1 = test_results['无所有三个模块']['avg_metrics'].get('f1_score', 0)
            
            print(f"\n{test_name}测试集:")
            print(f"  完整模型 F1: {full_f1:.4f}")
            print(f"  基线模型 F1: {baseline_f1:.4f}")
            
            if baseline_f1 > 0:
                improvement = (full_f1 - baseline_f1) / baseline_f1 * 100
                print(f"  三个模块整体提升: {improvement:.1f}%")
            
            # per-module contribution
            print("  各模块贡献:")
            for variant_name in ['无光照不变模块', '无反射增强模块', '无增强注意力模块']:
                if variant_name in test_results:
                    variant_f1 = test_results[variant_name]['avg_metrics'].get('f1_score', 0)
                    if full_f1 > 0:
                        contribution = (full_f1 - variant_f1) / full_f1 * 100
                        module_name = variant_name.replace('无', '')
                        print(f"    {module_name}: {contribution:.1f}%")
    
    print(f"\n所有结果已保存到: {args.output_dir}/")
    print(f"1. ablation_report.txt - 文本报告")
    print(f"2. detailed_results.json - 详细结果")
    print(f"3. *_ablation_results.png - 柱状图")
    print(f"4. *_radar_chart.png - 雷达图")


if __name__ == '__main__':
    main()
