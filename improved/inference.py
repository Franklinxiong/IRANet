"""
Batch inference for the improved model.

Input:  images under test_data/original.
Output: detection results under improved/result.
Weights: improved/models/best.pth.

Usage:
python improved/inference.py
python improved/inference.py --input PATH
"""

import os
import sys
import cv2
import torch
import numpy as np
import time
import argparse
from torchvision import transforms
from torch.autograd import Variable

# make the project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaderd import RescaleT, ToTensorLab
from improved.model import ImprovedGlassNet


def normPRED(d):
    """
    Normalise predictions to [0, 1].
    """
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi + 1e-7)
    return dn


def preprocess_image(image_path, target_size=384):
    """
    Preprocess an image to match the model input.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # remember the original size
    original_h, original_w = image.shape[:2]

    # resize to the model input
    image_resized = cv2.resize(image_rgb, (target_size, target_size))

    # scale to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0

    return image_normalized, image_rgb, (original_h, original_w)


def prepare_input_tensor(image_normalized):
    """
    Turn the preprocessed image into a model input tensor.
    """
    # same transforms as training
    transform = transforms.Compose([RescaleT(384), ToTensorLab(flag=0)])

    # dict mimicking the dataset samples
    h, w = image_normalized.shape[:2]
    dummy_label = np.zeros((h, w, 1), dtype=np.float32)
    dummy_depth = np.zeros((h, w), dtype=np.float32)
    dummy_depth_missing = np.ones((h, w), dtype=np.float32)

    sample = {
        'image': image_normalized,
        'label': dummy_label,
        'depth': dummy_depth,
        'depthmissing': dummy_depth_missing
    }

    # apply transforms
    sample = transform(sample)

    # image tensor with a batch axis
    image_tensor = sample['image'].unsqueeze(0).type(torch.FloatTensor)

    return image_tensor


def save_results(image_path, original_image, pred_mask, output_dir):
    """
    Save the inference result.
    """
    os.makedirs(output_dir, exist_ok=True)

    # filename without extension
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]

    # prediction to numpy
    pred_np = pred_mask.squeeze().cpu().data.numpy()

    # resize back to the original size
    h, w = original_image.shape[:2]
    pred_resized = cv2.resize(pred_np, (w, h))

    # output an RGB overlay: the prediction as a JET heatmap blended onto the image
    mask_img = (pred_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(
        cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), 0.7,
        heatmap, 0.3, 0
    )
    overlay_output_path = os.path.join(output_dir, f"{name_without_ext}.jpg")
    cv2.imwrite(overlay_output_path, overlay)

    return {'overlay': overlay_output_path}


def load_improved_model(weight_path, device):
    """
    Load the improved model's weights.
    """
    print(f"加载权重: {weight_path}")

    # load the checkpoint
    checkpoint = torch.load(weight_path, map_location=device)

    # build the model
    model = ImprovedGlassNet(backbone_path=None)

    # the checkpoint may come in slightly different layouts
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # drop keys that do not match
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape}

    skipped = len(state_dict) - len(filtered)
    if skipped > 0:
        print(f"  跳过 {skipped} 个不匹配的键")

    model_dict.update(filtered)
    model.load_state_dict(model_dict)

    return model


def batch_inference(input_dir, output_dir, use_cuda=True):
    """
    Batch inference main loop.
    """
    print("=" * 60)
    print("改进模型批量推理")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # check the input directory
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在: {input_dir}")
        return

    # list all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(input_dir, file))

    if not image_files:
        print(f"错误：在 {input_dir} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 张图片")

    # weight path
    weight_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "improved/models/best.pth")

    # check the weights exist
    if not os.path.exists(weight_path):
        print(f"错误：权重文件不存在: {weight_path}")
        print("请将 best.pth 放入 improved/models/ 目录")
        return

    # pick the device
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("设备: CUDA")
    else:
        device = torch.device('cpu')
        print("设备: CPU")
        use_cuda = False

    # load the model
    print("\n1. 加载改进模型...")
    model = load_improved_model(weight_path, device)
    model.to(device)
    model.eval()

    # model details
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   参数量: {total_params:.2f}M")

    # batch inference
    print("\n2. 开始批量推理...")
    total_time = 0
    results = []

    for i, image_path in enumerate(image_files):
        basename = os.path.basename(image_path)
        print(f"  [{i+1}/{len(image_files)}] {basename}", end=" ", flush=True)

        try:
            start_time = time.time()

            # preprocess
            image_normalized, image_rgb, _ = preprocess_image(image_path)

            # build the input tensor
            inputs_test = prepare_input_tensor(image_normalized)

            # move to device
            if use_cuda:
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            # forward (4 outputs, use the finest p1)
            with torch.no_grad():
                p1, p2, p3, p4 = model(inputs_test)

            # take the finest prediction
            pred = p1[:, 0, :, :]

            # post-process
            pred = torch.sigmoid(pred)
            pred = normPRED(pred)

            inference_time = time.time() - start_time
            total_time += inference_time

            # save the result
            save_results(image_path, image_rgb, pred, output_dir)

            print(f"{inference_time*1000:.0f}ms")

            results.append({
                'image': image_path,
                'time': inference_time
            })

        except Exception as e:
            print(f"失败: {e}")
            import traceback
            traceback.print_exc()

    # print a short summary
    print()
    print("=" * 60)
    if results:
        print(f"完成！{len(results)}张, {total_time:.2f}s, "
              f"{total_time/len(results)*1000:.0f}ms/张")
    else:
        print("未处理任何图片")
    print(f"输出: {output_dir}")
    print("=" * 60)

    return results


def main():
    """
    Entry point.
    """
    parser = argparse.ArgumentParser(description='改进模型批量推理')
    parser.add_argument('--input', type=str, default='test_data/original',
                        help='输入图像目录路径 (默认: test_data/original)')
    parser.add_argument('--output', type=str, default='improved/result',
                        help='输出目录路径 (默认: improved/result)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA（即使可用）')

    args = parser.parse_args()

    batch_inference(
        input_dir=args.input,
        output_dir=args.output,
        use_cuda=not args.no_cuda
    )


if __name__ == "__main__":
    main()