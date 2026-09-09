"""
RGB-only model optimisation script.

Loads an existing best.pth checkpoint, adapts it to the RGB-only
architecture and fine-tunes it on RGB-only data, with the goal of improving
performance under low-light and weakly reflective conditions.

Highlights:
1. Load an existing best.pth.
2. Adapt its weights to the RGB-only model.
3. Fine-tune on an RGB-only dataset.
4. Target low-light and weakly reflective scenes.
5. Runs on Apple Silicon Macs and Windows (CUDA 12.6).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import glob
import numpy as np
import argparse
import json
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend so training is not blocked

# project imports
from model_rgb_only import RGBOnlyGlassNet
from model_depth import GlassNet
from train_rgb_only import RGBOnlyDataset, enhanced_loss, muti_bce_loss_fusion, detect_environment
from data_loaderd import SalObjDataset

# try the shared metrics module, else fall back to local ones
try:
    import metrics
    HAS_METRICS = True
except ModuleNotFoundError as e:
    print(f"警告: 无法导入metrics模块: {e}")
    print("将使用备用实现来计算MAE和BER")
    HAS_METRICS = False
    
    # local fallback implementations
    import numpy as np
    
    class MetricsStub:
        @staticmethod
        def compute_mae(predict_mask, gt_mask):
            """
            Mean absolute error (MAE).

            Input should be float32 arrays with values in [0, 1].
            """
            # make sure the input is a numpy array
            if not isinstance(predict_mask, np.ndarray):
                predict_mask = np.array(predict_mask)
            if not isinstance(gt_mask, np.ndarray):
                gt_mask = np.array(gt_mask)
            
            # shapes must agree
            assert predict_mask.shape == gt_mask.shape, "预测和真实掩码形状不匹配"
            
            # MAE
            mae_value = np.mean(np.abs(predict_mask - gt_mask)).item()
            return mae_value
        
        @staticmethod
        def compute_ber(predict_mask, gt_mask):
            """
            Balanced error rate (BER).

            Input should be uint8 arrays with values 0 or 1.
            """
            # make sure the input is a numpy array
            if not isinstance(predict_mask, np.ndarray):
                predict_mask = np.array(predict_mask)
            if not isinstance(gt_mask, np.ndarray):
                gt_mask = np.array(gt_mask)
            
            # shapes must agree
            assert predict_mask.shape == gt_mask.shape, "预测和真实掩码形状不匹配"
            
            # inputs must be binary
            predict_mask = (predict_mask > 0.5).astype(np.float32)
            gt_mask = (gt_mask > 0.5).astype(np.float32)
            
            # true/false positives and negatives
            TP = np.sum(np.logical_and(predict_mask == 1, gt_mask == 1)).astype(np.float32)
            TN = np.sum(np.logical_and(predict_mask == 0, gt_mask == 0)).astype(np.float32)
            FP = np.sum(np.logical_and(predict_mask == 1, gt_mask == 0)).astype(np.float32)
            FN = np.sum(np.logical_and(predict_mask == 0, gt_mask == 1)).astype(np.float32)
            
            # count positive and negative pixels
            N_p = TP + FN
            N_n = TN + FP
            
            # guard against division by zero
            if N_p == 0 or N_n == 0:
                return 1.0  # worst case: every pixel wrong
            
            # BER = 1 - 0.5 * (TP/N_p + TN/N_n)
            ber_value = 1.0 - 0.5 * (TP / N_p + TN / N_n)
            return ber_value.item()
    
    # dummy object
    metrics = MetricsStub()


def detect_model_type_from_weights(weight_path):
    """
    Guess the model type from a weight file.

    Arguments:
    weight_path: path to the weights.

    Returns:
    str: one of 'rgbd', 'rgb_only', 'uav', 'gdnet', 'unknown'.
    """
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
        weight_keys = list(checkpoint.keys())
        
        # classify the architecture from the keys
        depth_keys = [k for k in weight_keys if 'depth' in k.lower()]
        rgb_only_keys = [k for k in weight_keys if 'rgb_contrast' in k.lower() or 'illumination_invariant' in k.lower()]
        uav_keys = [k for k in weight_keys if 'outdoor' in k.lower() or 'adaptive' in k.lower()]
        gdnet_keys = [k for k in weight_keys if 'h5_conv' in k.lower() or 'h_fusion' in k.lower() or 'l_fusion' in k.lower()]
        
        if len(gdnet_keys) > 0:
            return 'gdnet'
        elif len(uav_keys) > 0:
            return 'uav'
        elif len(depth_keys) > 0:
            return 'rgbd'
        elif len(rgb_only_keys) > 0:
            return 'rgb_only'
        else:
            # fall back to guessing from the key names
            if any('depth' in k.lower() for k in weight_keys):
                return 'rgbd'
            elif any('outdoor' in k.lower() for k in weight_keys):
                return 'uav'
            elif any('h5_conv' in k.lower() for k in weight_keys) or any('h_fusion' in k.lower() for k in weight_keys):
                return 'gdnet'
            else:
                return 'rgb_only'
    except:
        return 'rgb_only'  # default


def create_generic_model(weight_path, device):
    """
    Build a generic model that can load weights of an unknown architecture.

    Arguments:
    weight_path: path to the weights.
    device: target device.

    Returns:
    model: a generic model instance.
    """
    print("创建通用模型以加载未知架构权重...")
    
    # load the weights to inspect the architecture
    checkpoint = torch.load(weight_path, map_location=device)
    weight_keys = list(checkpoint.keys())
    
    # build a lightweight generic model
    class GenericGlassNet(nn.Module):
        def __init__(self):
            super(GenericGlassNet, self).__init__()
            
            # backbone (same as RGBOnlyGlassNet)
            from backbone.resnext.resnext101_regular import ResNeXt101
            resnext = ResNeXt101(None)
            self.layer0 = resnext.layer0
            self.layer1 = resnext.layer1
            self.layer2 = resnext.layer2
            self.layer3 = resnext.layer3
            self.layer4 = resnext.layer4
            
            # channel reduction
            self.cr4 = nn.Sequential(nn.Conv2d(2048, 512, 1, 1, 0), nn.BatchNorm2d(512), nn.ReLU())
            self.cr3 = nn.Sequential(nn.Conv2d(1024, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU())
            self.cr2 = nn.Sequential(nn.Conv2d(512, 128, 1, 1, 0), nn.BatchNorm2d(128), nn.ReLU())
            self.cr1 = nn.Sequential(nn.Conv2d(256, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU())
            
            # upsampling
            self.up_4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU())
            self.up_3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU())
            self.up_2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU())
            self.up_1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU())
            
            # prediction head
            self.layer4_predict = nn.Conv2d(256, 1, 3, 1, 1)
            self.layer3_predict = nn.Conv2d(128, 1, 3, 1, 1)
            self.layer2_predict = nn.Conv2d(64, 1, 3, 1, 1)
            self.layer1_predict = nn.Conv2d(32, 1, 3, 1, 1)
        
        def forward(self, x):
            # import F
            import torch.nn.functional as F
            
            # backbone features
            layer0 = self.layer0(x)
            layer1 = self.layer1(layer0)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)
            
            # channel reduction
            rgb_layer4 = self.cr4(layer4)
            rgb_layer3 = self.cr3(layer3)
            rgb_layer2 = self.cr2(layer2)
            rgb_layer1 = self.cr1(layer1)
            
            # multi-scale decoder
            up4 = self.up_4(rgb_layer4)
            layer4_predict = self.layer4_predict(up4)
            
            up3 = self.up_3(rgb_layer3 + up4)
            layer3_predict = self.layer3_predict(up3)
            
            up2 = self.up_2(rgb_layer2 + up3)
            layer2_predict = self.layer2_predict(up2)
            
            up1 = self.up_1(rgb_layer1 + up2)
            layer1_predict = self.layer1_predict(up1)
            
            # upsample to the original size
            layer4_predict = F.interpolate(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
            layer3_predict = F.interpolate(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
            layer2_predict = F.interpolate(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
            layer1_predict = F.interpolate(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
            
            return layer1_predict, layer2_predict, layer3_predict, layer4_predict
    
    # create the model
    model = GenericGlassNet()
    
    # report what the weight keys suggest
    print(f"权重文件包含 {len(weight_keys)} 个键")
    print("前10个权重键名:")
    for i, key in enumerate(weight_keys[:10]):
        print(f"  {i+1}. {key}")
    
    if len(weight_keys) > 10:
        print(f"  ... 还有 {len(weight_keys) - 10} 个键")
    
    return model


def load_model_with_auto_detection(weight_path, device):
    """
    Automatically detect the architecture and load the model.

    Arguments:
    weight_path: path to the weights.
    device: target device.

    Returns:
    tuple: (model, model_type).
    """
    print(f"加载权重: {weight_path}")
    
    # detect the model type
    model_type = detect_model_type_from_weights(weight_path)
    
    # create the matching model
    if model_type == 'rgbd':
        print("检测到RGBD模型权重，加载GlassNet...")
        model = GlassNet()
    elif model_type == 'uav':
        print("检测到UAV模型权重，加载SimplifiedUAVGlassNet...")
        from UAV.model_uav import SimplifiedUAVGlassNet
        model = SimplifiedUAVGlassNet(backbone_path=None)
    elif model_type == 'gdnet':
        print("检测到GDNet模型权重，加载GDNet...")
        from gdnet import GDNet
        model = GDNet(backbone_path=None)
    elif model_type == 'unknown':
        print("检测到未知模型架构，尝试使用通用加载器...")
        model = create_generic_model(weight_path, device)
    else:  # rgb_only
        print("检测到RGB-only模型权重，加载RGBOnlyGlassNet...")
        model = RGBOnlyGlassNet(backbone_path=None)
    
    # load the weights
    try:
        checkpoint = torch.load(weight_path, map_location=device)
        
        if model_type == 'unknown':
            # unknown models: load with strict=False
            model.load_state_dict(checkpoint, strict=False)
            print(f"权重加载成功 (部分加载，strict=False)")
        else:
            # known models: prefer a strict load
            try:
                model.load_state_dict(checkpoint)
                print(f"权重加载成功")
            except Exception as e:
                print(f"严格加载失败: {e}")
                print("尝试使用strict=False加载...")
                model.load_state_dict(checkpoint, strict=False)
                print(f"权重加载成功 (部分加载，strict=False)")
                
    except Exception as e:
        print(f"权重加载失败: {e}")
        print("使用随机初始化的模型")
    
    return model, model_type


def load_best_pth_weights(model, best_pth_path, device):
    """
    Load best.pth and adapt it to the RGB-only architecture.

    Arguments:
    model: the RGB-only model.
    best_pth_path: path to best.pth.
    device: cpu/cuda.

    Returns:
    model: the model with adapted weights.
    loaded_keys: keys successfully loaded.
    missing_keys: keys expected but absent.
    unexpected_keys: keys present but unknown.
    """
    print(f"正在加载best.pth权重: {best_pth_path}")
    
    # check the file exists
    if not os.path.exists(best_pth_path):
        raise FileNotFoundError(f"找不到best.pth文件: {best_pth_path}")
    
    # load the raw weights
    original_state_dict = torch.load(best_pth_path, map_location=device)
    print(f"原始权重包含 {len(original_state_dict)} 个键")
    
    # state dict of the current model
    model_state_dict = model.state_dict()
    
    # new state dict, keeping only the compatible weights
    new_state_dict = {}
    loaded_keys = []
    missing_keys = []
    unexpected_keys = []
    
    # 1. start with the backbone keys (most likely to line up)
    backbone_prefixes = ['layer0.', 'layer1.', 'layer2.', 'layer3.', 'layer4.']
    
    for key in model_state_dict.keys():
        # is there a matching original key?
        if key in original_state_dict:
            # exact name match, load directly
            new_state_dict[key] = original_state_dict[key]
            loaded_keys.append(key)
        else:
            # hunt for a compatible source key
            found = False
            
            # for backbone keys, look for a matching original name
            for prefix in backbone_prefixes:
                if key.startswith(prefix):
                    # try the same name first
                    if key in original_state_dict:
                        new_state_dict[key] = original_state_dict[key]
                        loaded_keys.append(key)
                        found = True
                        break
            
            if not found:
                missing_keys.append(key)
    
    # which original keys were never used
    for key in original_state_dict.keys():
        if key not in loaded_keys and key not in model_state_dict:
            unexpected_keys.append(key)
    
    print(f"成功加载 {len(loaded_keys)}/{len(model_state_dict)} 个权重")
    print(f"缺失 {len(missing_keys)} 个权重")
    print(f"有 {len(unexpected_keys)} 个意外的权重")
    
    # load the adapted weights
    model.load_state_dict(new_state_dict, strict=False)
    
    return model, loaded_keys, missing_keys, unexpected_keys


def create_optimizer(model, lr=1e-4):
    """
    Create the optimizer.

    Arguments:
    model: the model to optimise.
    lr: learning rate.

    Returns:
    optimizer: the optimizer.
    """
    # per-group learning rates
    backbone_params = []
    new_module_params = []
    
    for name, param in model.named_parameters():
        if 'layer0' in name or 'layer1' in name or 'layer2' in name or 'layer3' in name or 'layer4' in name:
            backbone_params.append(param)
        else:
            new_module_params.append(param)
    
    # backbone: small LR for fine-tuning
    # new modules: normal LR, trained from scratch
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': lr * 0.1},  # backbone at 0.1x
        {'params': new_module_params, 'lr': lr}       # new modules at 1x
    ], lr=lr)
    
    return optimizer


def adjust_lr(optimizer, epoch, init_lr, decay_rate=0.9, decay_epoch=10):
    """
    Step the learning rate.

    Arguments:
    optimizer: the optimizer.
    epoch: current epoch.
    init_lr: initial learning rate.
    decay_rate: decay factor.
    decay_epoch: decay interval.
    """
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs, model_type='rgb_only'):
    """
    Train for one epoch.

    Arguments:
    model: the model.
    dataloader: training loader.
    optimizer: the optimizer.
    device: device.
    epoch: current epoch.
    total_epochs: total number of epochs.
    model_type: one of 'rgbd', 'rgb_only'.

    Returns:
    avg_loss: average loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for i, data in enumerate(dataloader):
        # fetch a batch
        inputs, labels = data['image'], data['label']
        
        # move to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # forward depends on the model type
        if model_type == 'rgbd':
            # RGB-D models expect a depth input
            depth = data['depth'].to(device)
            depth_missing = data['depthmissing'].to(device)
            
            # GlassNet returns 8 outputs (4 RGB, 4 depth)
            d1, d2, d3, d4, depth1, depth2, depth3, depth4 = model(inputs, depth, depth_missing)
            
            # only the RGB outputs are used for the loss
            _, loss = muti_bce_loss_fusion(d1, d2, d3, d4, labels)
        else:
            # RGB-only model
            d1, d2, d3, d4 = model(inputs)
            
            # multi-scale loss via muti_bce_loss_fusion
            # note: it returns the last-decoder loss and the total
            _, loss = muti_bce_loss_fusion(d1, d2, d3, d4, labels)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # bookkeeping
        total_loss += loss.item()
        num_batches += 1
        
        # progress print
        if i % 10 == 0:
            print(f'Epoch [{epoch}/{total_epochs}], Batch [{i}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device, model_type='rgb_only'):
    """
    Validation pass.

    Arguments:
    model: the model.
    dataloader: validation loader.
    device: device.
    model_type: one of 'rgbd', 'rgb_only'.

    Returns:
    avg_loss: average loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data in dataloader:
            # fetch a batch
            inputs, labels = data['image'], data['label']
            
            # move to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward depends on the model type
            if model_type == 'rgbd':
                # RGB-D models expect a depth input
                depth = data['depth'].to(device)
                depth_missing = data['depthmissing'].to(device)
                
                # GlassNet returns 8 outputs (4 RGB, 4 depth)
                d1, d2, d3, d4, depth1, depth2, depth3, depth4 = model(inputs, depth, depth_missing)
                
                # only the RGB outputs are used for the loss
                _, loss = muti_bce_loss_fusion(d1, d2, d3, d4, labels)
            else:
                # RGB-only model
                d1, d2, d3, d4 = model(inputs)
                
                # multi-scale loss via muti_bce_loss_fusion
                # note: it returns the last-decoder loss and the total
                _, loss = muti_bce_loss_fusion(d1, d2, d3, d4, labels)
            
            # bookkeeping
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def test_model(model, test_dataloader, device, dataset_name="test", model_type='rgb_only'):
    """
    Evaluate the model and report F1, mIoU, MAE and BER.

    Arguments:
    model: the model.
    test_dataloader: test loader.
    device: device.
    dataset_name: name used in the report.
    model_type: one of 'rgbd', 'rgb_only', 'uav', 'gdnet'.

    Returns:
    metrics: dictionary of evaluation metrics.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_predictions_raw = []  # raw predictions before binarisation
    total_loss = 0.0
    num_batches = 0
    
    print(f"  开始测试，数据加载器长度: {len(test_dataloader)}")
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            if batch_idx % 10 == 0:
                print(f"  处理批次 {batch_idx}/{len(test_dataloader)}...")
            
            # fetch a batch
            inputs, labels = data['image'], data['label']
            
            # move to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward depends on the model type
            if model_type == 'rgbd':
                # RGB-D models expect a depth input
                depth = data['depth'].to(device)
                depth_missing = data['depthmissing'].to(device)
                
                # GlassNet returns 8 outputs (4 RGB, 4 depth)
                d1, d2, d3, d4, depth1, depth2, depth3, depth4 = model(inputs, depth, depth_missing)
                
                # only the RGB outputs are used for the loss
                _, loss = muti_bce_loss_fusion(d1, d2, d3, d4, labels)
                
                # take the finest RGB decoder output as the prediction
                predictions_raw = torch.sigmoid(d4)  # sigmoid activation
            elif model_type == 'gdnet':
                # GDNet model
                h_predict, l_predict, final_predict = model(inputs)
                
                # loss on the final prediction
                # GDNet returns 3 outputs, so the loss path is adapted
                # use the final prediction for the loss
                import torch.nn.functional as F
                loss = F.binary_cross_entropy(final_predict, labels)
                
                # use the final prediction
                predictions_raw = final_predict
            else:
                # RGB-only or UAV model
                d1, d2, d3, d4 = model(inputs)
                
                # compute the loss
                _, loss = muti_bce_loss_fusion(d1, d2, d3, d4, labels)
                
                # finest decoder output as the prediction
                predictions_raw = torch.sigmoid(d4)  # sigmoid activation
            
            predictions = (predictions_raw > 0.5).float()  # binarise
            
            total_loss += loss.item()
            num_batches += 1
            
            # collect predictions and labels for scoring
            all_predictions.append(predictions.cpu().numpy())
            all_predictions_raw.append(predictions_raw.cpu().numpy())  # keep the raw predictions for MAE
            all_labels.append(labels.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"    批次 {batch_idx} 完成，损失: {loss.item():.4f}")
    
    print(f"  测试完成，处理了 {num_batches} 个批次")
    
    # average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # concatenate all batches
    if all_predictions:
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_predictions_raw = np.concatenate(all_predictions_raw, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # flatten the arrays before scoring
        predictions_flat = all_predictions.flatten()
        predictions_raw_flat = all_predictions_raw.flatten()
        labels_flat = all_labels.flatten()
        
        # sklearn metrics need integer labels
        # because sklearn metrics require integer labels
        predictions_flat_int = (predictions_flat > 0.5).astype(np.int32)
        labels_flat_int = (labels_flat > 0.5).astype(np.int32)
        
        # score (F1, IoU)
        precision = precision_score(labels_flat_int, predictions_flat_int, zero_division=0)
        recall = recall_score(labels_flat_int, predictions_flat_int, zero_division=0)
        f1 = f1_score(labels_flat_int, predictions_flat_int, zero_division=0)
        iou = jaccard_score(labels_flat_int, predictions_flat_int, zero_division=0)
        
        # MAE and BER from the metrics module
        # MAE uses the raw (0-1) predictions, not the binarised ones
        mae_value = metrics.compute_mae(predictions_raw_flat.astype(np.float32), labels_flat.astype(np.float32))
        
        # BER needs binarised predictions and labels
        ber_value = metrics.compute_ber(predictions_flat_int.astype(np.uint8), labels_flat_int.astype(np.uint8))
        
        metrics_dict = {
            'dataset': dataset_name,
            'loss': avg_loss,
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'iou': float(iou),
            'mae': float(mae_value),
            'ber': float(ber_value),
            'num_samples': len(all_predictions),
            'model_type': model_type
        }
    else:
        metrics_dict = {
            'dataset': dataset_name,
            'loss': avg_loss,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'iou': 0.0,
            'mae': 0.0,
            'ber': 0.0,
            'num_samples': 0,
            'model_type': model_type
        }
    
    return metrics_dict


def create_test_dataset(test_dir, transform):
    """
    Create the test dataset.

    Arguments:
    test_dir: test directory (contains images and masks subdirs).
    transform: data transforms.

    Returns:
    test_dataset: the dataset.
    test_files: list of test files.
    """
    # verify the directory layout
    image_dir = os.path.join(test_dir, 'images')
    mask_dir = os.path.join(test_dir, 'masks')
    
    if not os.path.exists(image_dir):
        # try an alternative layout
        image_dir = test_dir
        mask_dir = test_dir.replace('images', 'masks') if 'images' in test_dir else test_dir
    
    if not os.path.exists(image_dir):
        print(f"警告: 测试图像目录不存在: {image_dir}")
        return None, []
    
    # collect the image files
    image_files = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))
    
    if len(image_files) == 0:
        print(f"警告: 测试目录中没有找到图像文件: {image_dir}")
        return None, []
    
    # images and masks must line up
    valid_image_files = []
    valid_mask_files = []
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(img_name)[0]
        
        # try several mask naming conventions
        mask_candidates = [
            os.path.join(mask_dir, f"{name_without_ext}_mask.png"),
            os.path.join(mask_dir, f"{name_without_ext}.png"),
            os.path.join(mask_dir, f"{name_without_ext}_mask.jpg"),
            os.path.join(mask_dir, f"{name_without_ext}.jpg"),
        ]
        
        mask_path = None
        for candidate in mask_candidates:
            if os.path.exists(candidate):
                mask_path = candidate
                break
        
        if mask_path and os.path.exists(mask_path):
            valid_image_files.append(img_path)
            valid_mask_files.append(mask_path)
        else:
            print(f"警告: 测试图像 {img_name} 没有对应的掩码文件")
    
    if len(valid_image_files) == 0:
        print(f"错误: 没有找到有效的测试图像-掩码对")
        return None, []
    
    print(f"找到 {len(valid_image_files)} 个测试样本")
    
    # build the dataset
    test_dataset = RGBOnlyDataset(
        img_name_list=valid_image_files,
        lbl_name_list=valid_mask_files,
        dep_name_list=[],  # no depth data needed
        transform=transform
    )
    
    return test_dataset, valid_image_files


class RGBOnlyToTensor(object):
    """
    ToTensor variant for RGB-only inputs that always yields float32.

    Needed because MPS devices do not support float64.
    """
    def __init__(self, flag=0):
        self.flag = flag
    
    def __call__(self, sample):
        # RGB images and labels only
        image, label = sample['image'], sample['label']
        
        # enforce float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
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
        
        # label handling
        if np.max(label) < 1e-6:
            tmpLbl = label
        else:
            tmpLbl = label / np.max(label)
        
        # (H, W, C) -> (C, H, W)
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = tmpLbl.transpose((2, 0, 1))
        
        # to a float32 tensor
        return {
            'image': torch.from_numpy(tmpImg).float(),
            'label': torch.from_numpy(tmpLbl).float(),
            'depth': torch.zeros(1, tmpImg.shape[1], tmpImg.shape[2]).float(),  # empty depth
            'depthmissing': torch.zeros(1, tmpImg.shape[1], tmpImg.shape[2]).float()  # empty depth-missing map
        }


def visualize_training(train_losses, val_losses, save_dir, title="训练过程可视化"):
    """
    Plot the training run.

    Arguments:
    train_losses: training losses.
    val_losses: validation losses.
    save_dir: where to save the figure.
    title: plot title.
    """
    # make the plotting dir
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # set up the figure
    plt.figure(figsize=(12, 8))
    
    # training loss curve
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2, marker='o', markersize=4)
    
    # validation loss curve
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2, marker='s', markersize=4)
    
    # polish the plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('损失', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(vis_dir, f'training_plot_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练可视化图表已保存: {plot_path}")
    
    # also dump the losses to CSV
    csv_path = os.path.join(vis_dir, f'training_data_{timestamp}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses if val_losses else [0]*len(train_losses)), 1):
            # val_loss can be None; handle it
            val_loss_value = val_loss if val_loss is not None else 0.0
            f.write(f"{i},{train_loss:.6f},{val_loss_value:.6f}\n")
    
    print(f"训练数据已保存: {csv_path}")
    
    return plot_path, csv_path


def main():
    """
    Entry point.
    """
    parser = argparse.ArgumentParser(description='RGB-only模型优化和测试')
    parser.add_argument('--data_dir', type=str, default='data/train/train', help='训练数据集目录（默认：data/train/train）')
    parser.add_argument('--best_pth', type=str, default='best.pth', help='best.pth文件路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练epoch数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_dir', type=str, default='optimized_models', help='模型保存目录')
    parser.add_argument('--val_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test_easy_dir', type=str, default='data/test/test/easy', help='easy测试集目录')
    parser.add_argument('--test_hard_dir', type=str, default='data/test/test/hard', help='hard测试集目录')
    parser.add_argument('--val_easy_dir', type=str, default='data/validation/validation/easy', help='easy验证集目录')
    parser.add_argument('--val_hard_dir', type=str, default='data/validation/validation/hard', help='hard验证集目录')
    parser.add_argument('--skip_training', action='store_true', help='跳过训练，直接测试现有模型')
    parser.add_argument('--test_model_path', type=str, default=None, help='要测试的模型路径（如果跳过训练）')
    parser.add_argument('--visualize', action='store_true', help='启用训练过程可视化')
    
    args = parser.parse_args()
    
    # pick a default test dir if none was given
    if args.test_easy_dir == 'data/test/test/easy' and not os.path.exists(args.test_easy_dir):
        # try a few likely paths
        possible_paths = [
            'data/test/easy',
            'test/easy',
            'easy'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                args.test_easy_dir = path
                break
    
    if args.test_hard_dir == 'data/test/test/hard' and not os.path.exists(args.test_hard_dir):
        possible_paths = [
            'data/test/hard',
            'test/hard',
            'hard'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                args.test_hard_dir = path
                break
    
    print("=" * 60)
    print("RGB-only模型优化")
    print("=" * 60)
    
    # detect the environment
    env = detect_environment()
    print("环境检测结果:")
    for key, value in env.items():
        print(f"  {key}: {value}")
    
    # device straight from detect_environment
    device = env['device']
    print(f"使用设备: {device}")
    
    # show CUDA details when available
    if device.type == 'cuda':
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    elif device.type == 'mps':
        print("MPS设备 (Mac M1/M2/M3/M4)")
    
    # check the dataset
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"警告: 数据集目录不存在: {data_dir}")
        print("请确保数据集已放置在正确位置")
        print("期望的数据集结构:")
        print("  data/train/train/images/  # 训练图像")
        print("  data/train/train/masks/   # 训练掩码")
        return
    
    # list images, matching the dataset structure
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录不存在: {image_dir}")
        return
    
    if not os.path.exists(mask_dir):
        print(f"错误: 掩码目录不存在: {mask_dir}")
        return
    
    # list all images
    image_files = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))
    print(f"找到 {len(image_files)} 个图像文件")
    
    if len(image_files) == 0:
        print("错误: 没有找到图像文件")
        return
    
    # images and masks must line up
    valid_image_files = []
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(img_name)[0]
        
        # masks follow the {name}_mask.png convention
        mask_path = os.path.join(mask_dir, f"{name_without_ext}_mask.png")
        
        if os.path.exists(mask_path):
            valid_image_files.append(img_path)
        else:
            print(f"警告: 图像 {img_name} 没有对应的掩码文件")
    
    image_files = valid_image_files
    print(f"有效图像-掩码对数量: {len(image_files)}")
    
    if len(image_files) == 0:
        print("错误: 没有找到有效的图像-掩码对")
        return
    
    # build the model
    print("\n创建RGB-only模型...")
    model = RGBOnlyGlassNet(backbone_path=None)  # no pretrained weights; best.pth is loaded next
    
    # load and adapt best.pth
    try:
        model, loaded_keys, missing_keys, unexpected_keys = load_best_pth_weights(
            model, args.best_pth, device
        )
        
        print("\n权重加载摘要:")
        print(f"  成功加载: {len(loaded_keys)} 个权重")
        print(f"  缺失权重: {len(missing_keys)} 个（新模块将从头训练）")
        print(f"  意外权重: {len(unexpected_keys)} 个（RGBD相关权重，将被忽略）")
        
        # report missing keys (expected for new modules)
        if missing_keys:
            print("\n缺失权重示例（新模块）:")
            for i, key in enumerate(missing_keys[:5]):
                print(f"  {key}")
            if len(missing_keys) > 5:
                print(f"  ... 还有 {len(missing_keys) - 5} 个")
        
    except Exception as e:
        print(f"加载best.pth权重失败: {e}")
        print("将使用随机初始化的模型")
    
    # move to device
    model = model.to(device)
    
    # set up the optimizer
    optimizer = create_optimizer(model, args.lr)
    
    # output dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    # training loop, unless --skip_train was set
    if not args.skip_training:
        # prepare the datasets
        print("\n准备数据集...")
        
        # split train/validation
        val_size = int(len(image_files) * args.val_split)
        train_files = image_files[val_size:]
        val_files = image_files[:val_size]
        
        print(f"训练集: {len(train_files)} 个图像")
        print(f"验证集: {len(val_files)} 个图像")
        
        # import the custom transforms
        from data_loaderd import RescaleT, RandomCrop, RandomHorizontalFlip
        
        # RGBOnlyToTensor keeps everything in float32
        train_transform = transforms.Compose([
            RescaleT(400),              # resize to 400px
            RandomCrop(384),            # random 384x384 crop
            RandomHorizontalFlip(0.5),  # flips with p=0.5
            RGBOnlyToTensor(flag=0)     # to float32 tensors
        ])
        
        val_transform = transforms.Compose([
            RescaleT(400),              # resize to 400px
            RandomCrop(384),            # centre crop to 384x384
            RGBOnlyToTensor(flag=0)     # to float32 tensors
        ])
        
        # dataset adapted to the actual layout
        def get_mask_path(img_path):
            """
            Return the mask path that corresponds to an image path.
            """
            img_name = os.path.basename(img_path)
            name_without_ext = os.path.splitext(img_name)[0]
            img_dir = os.path.dirname(img_path)
            mask_dir = img_dir.replace('images', 'masks')
            mask_path = os.path.join(mask_dir, f"{name_without_ext}_mask.png")
            return mask_path
        
        train_mask_files = [get_mask_path(f) for f in train_files]
        val_mask_files = [get_mask_path(f) for f in val_files]
        
        # drop images without a mask
        for i, mask_path in enumerate(train_mask_files):
            if not os.path.exists(mask_path):
                print(f"警告: 训练掩码文件不存在: {mask_path}")
        
        for i, mask_path in enumerate(val_mask_files):
            if not os.path.exists(mask_path):
                print(f"警告: 验证掩码文件不存在: {mask_path}")
        
        train_dataset = RGBOnlyDataset(
            img_name_list=train_files,
            lbl_name_list=train_mask_files,
            dep_name_list=[],  # no depth data needed
            transform=train_transform
        )
        
        val_dataset = RGBOnlyDataset(
            img_name_list=val_files,
            lbl_name_list=val_mask_files,
            dep_name_list=[],  # no depth data needed
            transform=val_transform
        )
        
        # build the data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(f"\n开始训练，共 {args.epochs} 个epoch...")
        print("-" * 60)
        
        best_val_loss = float('inf')
        
        # loss history
        train_losses = []
        val_losses = []
        
        for epoch in range(1, args.epochs + 1):
            # step the learning rate
            adjust_lr(optimizer, epoch, args.lr)
            
            # train one epoch
            train_loss = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
            
            # validate
            val_loss = validate(model, val_loader, device)
            
            # record losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # print the epoch summary
            print(f'Epoch [{epoch}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(args.save_dir, f'best_rgb_only_model.pth')
                torch.save(model.state_dict(), model_path)
                print(f'  保存最佳模型: {model_path} (Val Loss: {val_loss:.4f})')
            
            # checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f'  保存检查点: {checkpoint_path}')
                
                # optional visualisation every 10 epochs
                if args.visualize:
                    visualize_training(
                        train_losses, 
                        val_losses, 
                        args.save_dir, 
                        title=f"RGB-only玻璃检测训练过程 (Epoch {epoch})"
                    )
        
        print("-" * 60)
        print("训练完成!")
        
        # final visualisation after training
        if args.visualize:
            print("\n生成训练过程可视化...")
            plot_path, csv_path = visualize_training(
                train_losses, 
                val_losses, 
                args.save_dir, 
                title=f"RGB-only玻璃检测训练过程 (最终)"
            )
            print(f"  可视化图表: {plot_path}")
            print(f"  训练数据: {csv_path}")
        
        # save the final weights
        final_model_path = os.path.join(args.save_dir, 'final_rgb_only_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"最终模型已保存: {final_model_path}")
        
        # which model to evaluate
        test_model_path = final_model_path
    else:
        # skip training, test a given model
        if args.test_model_path:
            test_model_path = args.test_model_path
        else:
            # default to the best checkpoint
            test_model_path = os.path.join(args.save_dir, 'best_rgb_only_model.pth')
            if not os.path.exists(test_model_path):
                print(f"错误: 找不到测试模型: {test_model_path}")
                print("请提供有效的模型路径或运行训练")
                return
    
    
    # test phase: evaluate on validation and test sets
    
    print("\n" + "=" * 60)
    print("开始测试阶段")
    print("=" * 60)
    
    # transforms needed for testing
    from data_loaderd import RescaleT, RandomCrop, RandomHorizontalFlip
    
    # load the model to evaluate
    print(f"\n加载测试模型: {test_model_path}")
    if not os.path.exists(test_model_path):
        print(f"错误: 测试模型文件不存在: {test_model_path}")
        return
    
    # auto-detect and load the model
    test_model_instance, model_type = load_model_with_auto_detection(test_model_path, device)
    test_model_instance = test_model_instance.to(device)
    test_model_instance.eval()
    print(f"测试模型加载完成 (模型类型: {model_type})")
    
    # test transforms
    test_transform = transforms.Compose([
        RescaleT(400),              # resize to 400px
        RandomCrop(384),            # centre crop to 384x384
        RGBOnlyToTensor(flag=0)     # to float32 tensors
    ])
    
    # collect results
    all_test_results = []
    
    # 1. validation set (if present and training was not skipped)
    if not args.skip_training:
        print("\n" + "-" * 60)
        print("1. 验证集测试")
        print("-" * 60)

        validation_sets = [
            ('val_easy', args.val_easy_dir),
            ('val_hard', args.val_hard_dir)
        ]
        
        for val_name, val_dir in validation_sets:
            if os.path.exists(val_dir):
                print(f"\n在 {val_name} 验证集上测试: {val_dir}")
                val_dataset, val_files = create_test_dataset(val_dir, test_transform)
                
                if val_dataset and len(val_files) > 0:
                    print(f"创建数据加载器，批次大小: {args.batch_size}")
                    # fewer workers to avoid hangs, especially on Windows
                    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
                    print(f"数据加载器创建完成，开始测试...")
                    val_metrics = test_model(test_model_instance, val_loader, device, dataset_name=val_name, model_type=model_type)
                    all_test_results.append(val_metrics)
                    
                    # print the validation scores
                    print(f"  {val_name} 验证结果:")
                    print(f"    样本数量: {val_metrics['num_samples']}")
                    print(f"    损失: {val_metrics['loss']:.4f}")
                    print(f"    精确率: {val_metrics['precision']:.4f}")
                    print(f"    召回率: {val_metrics['recall']:.4f}")
                    print(f"    F1分数: {val_metrics['f1_score']:.4f}")
                    print(f"    IoU: {val_metrics['iou']:.4f}")
                    print(f"    MAE: {val_metrics['mae']:.4f}")
                    print(f"    BER: {val_metrics['ber']:.4f}")
                else:
                    print(f"  警告: {val_name} 验证集为空或无效")
            else:
                print(f"  警告: {val_name} 验证集目录不存在: {val_dir}")
    
    # 2. test set (easy and hard)
    print("\n" + "-" * 60)
    print("2. 测试集测试")
    print("-" * 60)
    
    test_sets = [
        ('test_easy', args.test_easy_dir),
        ('test_hard', args.test_hard_dir)
    ]
    
    for test_name, test_dir in test_sets:
        if os.path.exists(test_dir):
            print(f"\n在 {test_name} 测试集上测试: {test_dir}")
            test_dataset, test_files = create_test_dataset(test_dir, test_transform)
            
            if test_dataset and len(test_files) > 0:
                print(f"创建数据加载器，批次大小: {args.batch_size}")
                # fewer workers to avoid hangs, especially on Windows
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
                print(f"数据加载器创建完成，开始测试...")
                test_metrics = test_model(test_model_instance, test_loader, device, dataset_name=test_name, model_type=model_type)
                all_test_results.append(test_metrics)
                
                # print the test scores
                print(f"  {test_name} 测试结果:")
                print(f"    样本数量: {test_metrics['num_samples']}")
                print(f"    损失: {test_metrics['loss']:.4f}")
                print(f"    精确率: {test_metrics['precision']:.4f}")
                print(f"    召回率: {test_metrics['recall']:.4f}")
                print(f"    F1分数: {test_metrics['f1_score']:.4f}")
                print(f"    IoU: {test_metrics['iou']:.4f}")
                print(f"    MAE: {test_metrics['mae']:.4f}")
                print(f"    BER: {test_metrics['ber']:.4f}")
            else:
                print(f"  警告: {test_name} 测试集为空或无效")
        else:
            print(f"  警告: {test_name} 测试集目录不存在: {test_dir}")
    
    # 3. write the test report
    print("\n" + "-" * 60)
    print("3. 生成测试报告")
    print("-" * 60)
    
    # report dir
    report_dir = os.path.join(args.save_dir, 'test_reports')
    os.makedirs(report_dir, exist_ok=True)
    
    # report name with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"test_report_{timestamp}.txt"
    report_path = os.path.join(report_dir, report_filename)
    
    # write the report file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("RGB-only玻璃检测模型测试报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试模型: {test_model_path}\n")
        f.write(f"模型类型: {model_type}\n")
        f.write(f"设备: {device}\n\n")
        
        f.write("测试参数:\n")
        f.write(f"  训练数据目录: {args.data_dir}\n")
        f.write(f"  测试批次大小: {args.batch_size}\n")
        f.write(f"  Easy测试集目录: {args.test_easy_dir}\n")
        f.write(f"  Hard测试集目录: {args.test_hard_dir}\n")
        f.write(f"  Easy验证集目录: {args.val_easy_dir}\n")
        f.write(f"  Hard验证集目录: {args.val_hard_dir}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("测试结果汇总\n")
        f.write("=" * 60 + "\n\n")
        
        # per-dataset results
        for result in all_test_results:
            f.write(f"数据集: {result['dataset']}\n")
            f.write(f"  模型类型: {result.get('model_type', 'unknown')}\n")
            f.write(f"  样本数量: {result['num_samples']}\n")
            f.write(f"  损失: {result['loss']:.4f}\n")
            f.write(f"  精确率: {result['precision']:.4f}\n")
            f.write(f"  召回率: {result['recall']:.4f}\n")
            f.write(f"  F1分数: {result['f1_score']:.4f}\n")
            f.write(f"  IoU: {result['iou']:.4f}\n")
            f.write(f"  MAE: {result['mae']:.4f}\n")
            f.write(f"  BER: {result['ber']:.4f}\n\n")
        
        # overall average if there are several datasets
        if len(all_test_results) > 0:
            avg_loss = sum(r['loss'] for r in all_test_results) / len(all_test_results)
            avg_precision = sum(r['precision'] for r in all_test_results) / len(all_test_results)
            avg_recall = sum(r['recall'] for r in all_test_results) / len(all_test_results)
            avg_f1 = sum(r['f1_score'] for r in all_test_results) / len(all_test_results)
            avg_iou = sum(r['iou'] for r in all_test_results) / len(all_test_results)
            
            f.write("=" * 60 + "\n")
            f.write("平均指标\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"平均损失: {avg_loss:.4f}\n")
            f.write(f"平均精确率: {avg_precision:.4f}\n")
            f.write(f"平均召回率: {avg_recall:.4f}\n")
            f.write(f"平均F1分数: {avg_f1:.4f}\n")
            f.write(f"平均IoU: {avg_iou:.4f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("测试完成\n")
        f.write("=" * 60 + "\n")
    
    print(f"测试报告已保存: {report_path}")
    
    # 4. print the summary
    print("\n" + "=" * 60)
    print("测试摘要")
    print("=" * 60)
    
    if len(all_test_results) > 0:
        print(f"测试了 {len(all_test_results)} 个数据集")
        
        # group results by dataset
        val_results = [r for r in all_test_results if r['dataset'].startswith('val_')]
        test_results = [r for r in all_test_results if r['dataset'].startswith('test_')]
        
        if val_results:
            print("\n验证集结果:")
            for result in val_results:
                model_type_str = f" ({result.get('model_type', 'unknown')})"
                print(f"  {result['dataset']}{model_type_str}: F1={result['f1_score']:.4f}, IoU={result['iou']:.4f}")
        
        if test_results:
            print("\n测试集结果:")
            for result in test_results:
                model_type_str = f" ({result.get('model_type', 'unknown')})"
                print(f"  {result['dataset']}{model_type_str}: F1={result['f1_score']:.4f}, IoU={result['iou']:.4f}")
        
        # grand average
        if len(all_test_results) > 1:
            avg_f1 = sum(r['f1_score'] for r in all_test_results) / len(all_test_results)
            avg_iou = sum(r['iou'] for r in all_test_results) / len(all_test_results)
            print(f"\n总体平均: F1={avg_f1:.4f}, IoU={avg_iou:.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
