import os
import cv2
import torch
import numpy as np
import time
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import inspect

from data_loaderd import RescaleT, ToTensorLab
from model_depth import GlassNet
from model_rgb_only import RGBOnlyGlassNet
#from UAV.model_uav import SimplifiedUAVGlassNet

# optional CRF refinement; skip it if unavailable
try:
    from misc import crf_refine
    CRF_AVAILABLE = True
    print("CRF细化功能可用")
except ImportError:
    CRF_AVAILABLE = False
    print("警告：pydensecrf模块未安装，CRF细化功能不可用")

def normPRED(d):
    """
    Normalise predictions to [0, 1].
    """
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def detect_model_type(model):
    """
    Detect the model type.

    Arguments:
    model: a PyTorch model instance.

    Returns:
    str: one of 'rgbd', 'rgb_only', 'uav', 'gdnet'.
    """
    # inspect the model class name
    model_class_name = model.__class__.__name__
    
    if 'UAV' in model_class_name or 'SimplifiedUAV' in model_class_name:
        return 'uav'
    elif 'RGBOnly' in model_class_name:
        return 'rgb_only'
    elif 'GDNet' in model_class_name:
        return 'gdnet'
    elif 'GlassNet' in model_class_name:
        # check how many arguments forward takes
        forward_params = inspect.signature(model.forward).parameters
        if len(forward_params) == 3:  # forward(self, x, depth, depth_missing)
            return 'rgbd'
        else:
            return 'rgb_only'
    else:
        # default: infer from the forward signature
        try:
            forward_params = inspect.signature(model.forward).parameters
            if len(forward_params) == 3:
                return 'rgbd'
            else:
                return 'rgb_only'
        except:
            return 'rgb_only'


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
    class GenericGlassNet(torch.nn.Module):
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
            self.cr4 = torch.nn.Sequential(torch.nn.Conv2d(2048, 512, 1, 1, 0), torch.nn.BatchNorm2d(512), torch.nn.ReLU())
            self.cr3 = torch.nn.Sequential(torch.nn.Conv2d(1024, 256, 1, 1, 0), torch.nn.BatchNorm2d(256), torch.nn.ReLU())
            self.cr2 = torch.nn.Sequential(torch.nn.Conv2d(512, 128, 1, 1, 0), torch.nn.BatchNorm2d(128), torch.nn.ReLU())
            self.cr1 = torch.nn.Sequential(torch.nn.Conv2d(256, 64, 1, 1, 0), torch.nn.BatchNorm2d(64), torch.nn.ReLU())
            
            # upsampling
            self.up_4 = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                torch.nn.Conv2d(512, 256, 3, 1, 1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU())
            self.up_3 = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                torch.nn.Conv2d(256, 128, 3, 1, 1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU())
            self.up_2 = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                torch.nn.Conv2d(128, 64, 3, 1, 1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU())
            self.up_1 = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                torch.nn.Conv2d(64, 32, 3, 1, 1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU())
            
            # prediction head
            self.layer4_predict = torch.nn.Conv2d(256, 1, 3, 1, 1)
            self.layer3_predict = torch.nn.Conv2d(128, 1, 3, 1, 1)
            self.layer2_predict = torch.nn.Conv2d(64, 1, 3, 1, 1)
            self.layer1_predict = torch.nn.Conv2d(32, 1, 3, 1, 1)
        
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


def load_model_with_auto_detection(weight_path, use_cuda=True):
    """
    Automatically detect and load the model.

    Arguments:
    weight_path: path to the weights.
    use_cuda: whether to use CUDA.

    Returns:
    tuple: (model, model_type).
    """
    print(f"加载权重: {weight_path}")
    
    # pick the device
    map_location = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    
    # load the weights first to learn the architecture
    try:
        # load without specifying a model first
        checkpoint = torch.load(weight_path, map_location=map_location)
        
        # read off the key names
        weight_keys = list(checkpoint.keys())
        
        # classify the architecture from the keys
        depth_keys = [k for k in weight_keys if 'depth' in k.lower()]
        rgb_only_keys = [k for k in weight_keys if 'rgb_contrast' in k.lower() or 'illumination_invariant' in k.lower()]
        uav_keys = [k for k in weight_keys if 'outdoor' in k.lower() or 'adaptive' in k.lower()]
        gdnet_keys = [k for k in weight_keys if 'h5_conv' in k.lower() or 'h_fusion' in k.lower() or 'l_fusion' in k.lower()]
        
        if len(gdnet_keys) > 0:
            print("检测到GDNet模型权重，加载GDNet...")
            from gdnet import GDNet
            model = GDNet(backbone_path=None)
            model_type = 'gdnet'
        elif len(uav_keys) > 0:
            print("检测到UAV模型权重，加载SimplifiedUAVGlassNet...")
            model = SimplifiedUAVGlassNet(backbone_path=None)
            model_type = 'uav'
        elif len(depth_keys) > 0:
            print("检测到RGBD模型权重，加载GlassNet...")
            model = GlassNet()
            model_type = 'rgbd'
        elif len(rgb_only_keys) > 0:
            print("检测到RGB-only模型权重，加载RGBOnlyGlassNet...")
            model = RGBOnlyGlassNet(backbone_path=None)
            model_type = 'rgb_only'
        else:
            # if the keys are ambiguous, try the candidate models
            print("无法通过权重键名判断模型类型，尝试加载为SimplifiedUAVGlassNet...")
            try:
                model = SimplifiedUAVGlassNet(backbone_path=None)
                model.load_state_dict(checkpoint)
                model_type = 'uav'
                print("成功加载为SimplifiedUAVGlassNet (UAV模型)")
            except:
                print("SimplifiedUAVGlassNet加载失败，尝试加载为GlassNet...")
                try:
                    model = GlassNet()
                    model.load_state_dict(checkpoint)
                    model_type = 'rgbd'
                    print("成功加载为GlassNet (RGBD模型)")
                except:
                    print("GlassNet加载失败，尝试加载为RGBOnlyGlassNet...")
                    model = RGBOnlyGlassNet(backbone_path=None)
                    model.load_state_dict(checkpoint)
                    model_type = 'rgb_only'
                    print("成功加载为RGBOnlyGlassNet (RGB-only模型)")
        
        # unknown models load with strict=False
        if model_type == 'unknown':
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
        print(f"自动检测失败: {e}")
        print("使用默认的GlassNet模型...")
        model = GlassNet()
        model_type = 'rgbd'
        # try loading the weights
        try:
            model.load_state_dict(torch.load(weight_path, map_location=map_location))
        except:
            print("警告：权重加载失败，使用随机初始化")
    
    return model, model_type

def preprocess_image(image_path, target_size=384):
    """
    Preprocess an image to match the model input.
    """
    # load the image
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
    dummy_depth = np.zeros((h, w), dtype=np.float32)  # zero depth
    dummy_depth_missing = np.ones((h, w), dtype=np.float32)  # depth missing
    
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

def save_results(image_path, original_image, pred_mask, output_dir, use_crf=False):
    """
    Save the inference result.
    """
    # make sure the output dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    # filename without extension
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # prediction to numpy
    pred_np = pred_mask.squeeze().cpu().data.numpy()
    
    # resize back to the original size
    h, w = original_image.shape[:2]
    pred_resized = cv2.resize(pred_np, (w, h))
    
    # optional CRF refinement
    if use_crf and CRF_AVAILABLE:
        try:
            # prepare the CRF inputs
            input_for_crf = (original_image * 255).astype(np.uint8)
            pred_for_crf = (pred_resized * 255).astype(np.uint8)
            
            # run CRF refinement
            pred_resized = crf_refine(input_for_crf, pred_for_crf) / 255.0
            print(f"  已应用CRF细化")
        except Exception as e:
            print(f"  CRF细化失败: {e}")
    
    # output an RGB overlay: the prediction as a JET heatmap blended onto the image
    mask_img = (pred_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(
        cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), 0.7,
        heatmap, 0.3, 0
    )
    overlay_output_path = os.path.join(output_dir, f"{name_without_ext}.jpg")
    cv2.imwrite(overlay_output_path, overlay)
    
    return {
        'overlay': overlay_output_path
    }

def batch_inference(input_dir, output_dir, use_cuda=True, use_crf=False, model_path=None):
    """
    Batch inference main loop.
    """
    print(f"=== 批量推理开始 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"使用CUDA: {use_cuda}")
    print(f"使用CRF细化: {use_crf}")
    print()
    
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
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # weight path
    if model_path is None:
        weight_path = "./optimized_models/best_rgb_only_model.pth"
    else:
        weight_path = model_path
    
    # check the weights exist
    if not os.path.exists(weight_path):
        print(f"错误：权重文件不存在: {weight_path}")
        print("请确保权重文件路径正确，或使用 --model 参数指定权重文件")
        return
    
    # auto-detect and load the model
    print("\n1. 自动检测模型类型并加载...")
    net, model_type = load_model_with_auto_detection(weight_path, use_cuda)
    
    if use_cuda and torch.cuda.is_available():
        net.cuda()
        print("  使用CUDA加速")
    else:
        print("  使用CPU")
        use_cuda = False
    
    net.eval()
    
    # model details
    print(f"2. 模型信息:")
    print(f"   模型类型: {model_type}")
    print(f"   模型架构: {net.__class__.__name__}")
    
    # batch inference
    print(f"\n3. 开始批量推理 (模型类型: {model_type.upper()})...")
    total_time = 0
    results = []
    
    for i, image_path in enumerate(image_files):
        print(f"  [{i+1}/{len(image_files)}] 处理: {os.path.basename(image_path)}")
        
        try:
            # preprocess
            start_time = time.time()
            image_normalized, image_rgb, original_size = preprocess_image(image_path)
            
            # build the input tensor
            inputs_test = prepare_input_tensor(image_normalized)
            
            # input preparation depends on the model type
            if model_type == 'rgbd':
                # RGB-D models expect a depth input
                inputs_depth = torch.zeros((1, 1, 384, 384), dtype=torch.float32)
                inputs_dm = inputs_depth.clone()  # depth_missing mirrors depth here
                
                # move to GPU when available
                if use_cuda:
                    inputs_test = Variable(inputs_test.cuda())
                    inputs_depth = Variable(inputs_depth.cuda())
                    inputs_dm = Variable(inputs_dm.cuda())
                else:
                    inputs_test = Variable(inputs_test)
                    inputs_depth = Variable(inputs_depth)
                    inputs_dm = Variable(inputs_dm)
                
                # RGB-D forward
                with torch.no_grad():
                    d1, d2, d3, d4, depth1, depth2, depth3, depth4 = net(inputs_test, inputs_depth, inputs_dm)
                
                # take the first RGB output as the prediction
                pred = d1[:, 0, :, :]
                
            elif model_type == 'uav':
                # UAV models only need RGB
                # move to GPU when available
                if use_cuda:
                    inputs_test = Variable(inputs_test.cuda())
                else:
                    inputs_test = Variable(inputs_test)
                
                # UAV forward
                with torch.no_grad():
                    d1, d2, d3, d4 = net(inputs_test)
                
                # take the first output as the prediction
                pred = d1[:, 0, :, :]
                
            elif model_type == 'gdnet':
                # GDNet only needs RGB
                # move to GPU when available
                if use_cuda:
                    inputs_test = Variable(inputs_test.cuda())
                else:
                    inputs_test = Variable(inputs_test)
                
                # GDNet forward (3 outputs: h_predict, l_predict, final_predict)
                with torch.no_grad():
                    h_predict, l_predict, final_predict = net(inputs_test)
                
                # use the final prediction
                pred = final_predict
                
            elif model_type == 'unknown':
                # unknown models only need RGB
                # move to GPU when available
                if use_cuda:
                    inputs_test = Variable(inputs_test.cuda())
                else:
                    inputs_test = Variable(inputs_test)
                
                # forward for an unknown model
                with torch.no_grad():
                    d1, d2, d3, d4 = net(inputs_test)
                
                # take the first output as the prediction
                pred = d1[:, 0, :, :]
                
            else:  # rgb_only
                # RGB-only model
                # move to GPU when available
                if use_cuda:
                    inputs_test = Variable(inputs_test.cuda())
                else:
                    inputs_test = Variable(inputs_test)
                
                # RGB-only forward
                with torch.no_grad():
                    d1, d2, d3, d4 = net(inputs_test)
                
                # take the first output as the prediction
                pred = d1[:, 0, :, :]
            
            # post-process the prediction
            pred = torch.sigmoid(pred)
            pred = normPRED(pred)
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # save the result
            saved_files = save_results(image_path, image_rgb, pred, output_dir, use_crf=use_crf)
            
            results.append({
                'image': image_path,
                'time': inference_time,
                'files': saved_files
            })
            
            print(f"    完成！推理时间: {inference_time*1000:.1f}ms")
            
        except Exception as e:
            print(f"    错误处理 {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # print a short summary
    print("\n=== 批量推理完成 ===")
    print(f"总处理图像数: {len(results)}")
    print(f"总推理时间: {total_time:.2f}秒")
    print(f"平均每张图像推理时间: {total_time/len(results)*1000:.1f}ms")
    print(f"模型类型: {model_type.upper()}")
    print(f"输出目录: {output_dir}")
    print(f"生成的文件:")
    print(f"  - *.png: 二值检测掩码图（白色为检测目标，原始文件名.png）")
    
    return results

def main():
    """
    Entry point.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='批量玻璃检测推理（支持RGBD和RGB-only模型）')
    parser.add_argument('--input', type=str, default='test_data/original',
                       help='输入图像目录路径 (默认: test_data/original)')
    parser.add_argument('--output', type=str, default='test_data/detected',
                       help='输出目录路径 (默认: test_data/detected)')
    parser.add_argument('--model', type=str, default=None,
                       help='模型权重文件路径 (默认: ./optimized_models/best_rgb_only_model.pth)')
    parser.add_argument('--no_cuda', action='store_true',
                       help='禁用CUDA（即使可用）')
    parser.add_argument('--use_crf', action='store_true',
                       help='使用CRF细化（如果可用）')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='二值化阈值 (默认: 0.5)')
    
    args = parser.parse_args()
    
    # run batch inference
    batch_inference(
        input_dir=args.input,
        output_dir=args.output,
        use_cuda=not args.no_cuda,
        use_crf=args.use_crf,
        model_path=args.model
    )

if __name__ == "__main__":
    main()
