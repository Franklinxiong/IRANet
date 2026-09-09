"""
RGB-only glass detection model.

A glass segmentation network tuned for RGB-only input, with extra attention
to low illumination and weakly reflective glass. Runs on Apple Silicon Macs
and Windows (CUDA 12.6).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from backbone.resnext.resnext101_regular import ResNeXt101


class SELayer(nn.Module):
    """
    Selective attention layer (squeeze-and-excitation).

    Reweights feature channels so informative ones are kept and redundant
    ones are suppressed.

    Arguments:
    channel: number of input channels.
    reduction: channel reduction factor, default 4.
    num_context: number of context bins, default 8.
    """
    def __init__(self, channel, reduction=4, num_context=8):
        super(SELayer, self).__init__()
        self.channel = channel
        self.num_context = num_context
        
        # ensure the channel count divides evenly by num_context
        if channel % num_context != 0:
            # round the channel count up to a multiple of num_context
            self.channel = (channel // num_context + 1) * num_context
            print(f"警告: SELayer channel从{channel}调整为{self.channel}以确保能被{num_context}整除")
        
        self.context_channel = int(self.channel / num_context)
        
        # global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # context attention branch
        self.context_attention = nn.Sequential(
            nn.Conv2d(self.channel, self.channel // 2, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel // 2, num_context, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        # align mid_channels with num_context
        mid_channels = self.channel // reduction
        # keep mid_channels divisible by num_context
        if mid_channels % num_context != 0:
            mid_channels = (mid_channels // num_context + 1) * num_context
        
        # channel attention branch
        self.channel_attention = nn.Sequential(
            nn.Conv2d(self.channel, mid_channels, 1, 1, 0, groups=num_context, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, self.channel, 1, 1, 0, groups=num_context, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass.

        Arguments:
        x: input feature map [B, C, H, W].

        Returns:
        enhanced feature map [B, C, H, W].
        """
        b, c, _, _ = x.size()
        
        # global feature summary
        y = self.avg_pool(x)
        
        # combine the context and channel attention maps
        context_attention = self.context_attention(y)
        channel_attention = self.channel_attention(y)
        
        # expand context attention back to the full channel count
        context_attention = context_attention.repeat(1, 1, self.context_channel, 1)
        context_attention = context_attention.view(-1, self.channel, 1, 1)
        
        # blend the two attention maps
        attention = context_attention * channel_attention
        
        # re-weight the features with the combined attention
        return x * attention.expand_as(x)


class IlluminationInvariantModule(nn.Module):
    """
    Illumination-invariant feature extraction.

    Designed for low-light conditions: produces features that stay stable
    across illumination changes.

    Arguments:
    in_channels: number of input channels.
    out_channels: number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(IlluminationInvariantModule, self).__init__()
        
        # first conv: extract local features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # second conv: refine the features
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # attention to emphasise informative features
        self.attention = SELayer(out_channels)
        
    def forward(self, x):
        """
        Forward pass.

        Arguments:
        x: input feature map [B, C_in, H, W].

        Returns:
        illumination-invariant features [B, C_out, H, W].
        """
        # feature extraction
        x = self.conv1(x)
        
        # feature refinement
        x = self.conv2(x)
        
        # attention refinement
        x = self.attention(x)
        
        return x


class ReflectionEnhancementModule(nn.Module):
    """
    Reflection enhancement module.

    Built for weakly reflective glass and amplifies reflection cues.

    Arguments:
    channels: number of input/output channels.
    """
    def __init__(self, channels):
        super(ReflectionEnhancementModule, self).__init__()
        
        # first conv: reduce the channel count
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # second conv: restore the channel count
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels // 2, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # learnable scaling factor
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Forward pass.

        Arguments:
        x: input feature map [B, C, H, W].

        Returns:
        enhanced feature map [B, C, H, W].
        """
        # keep the residual branch
        residual = x
        
        # transform the features
        x = self.conv1(x)
        x = self.conv2(x)
        
        # residual connection scaled by gamma
        return residual + self.gamma * x


class EnhancedAttentionModule(nn.Module):
    """
    Enhanced attention module.

    Combines channel and spatial attention to improve feature selection.

    Arguments:
    channels: number of input channels.
    """
    def __init__(self, channels):
        super(EnhancedAttentionModule, self).__init__()
        
        # channel attention: weight useful channels
        self.channel_attention = SELayer(channels)
        
        # spatial attention: focus on informative locations
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3),  # 7x7 conv to capture a wider context
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass.

        Arguments:
        x: input feature map [B, C, H, W].

        Returns:
        attention-enhanced feature map [B, C, H, W].
        """
        # channel attention to boost useful channels
        x_channel = self.channel_attention(x)
        
        # spatial attention from average and max pooled features
        avg_out = torch.mean(x, dim=1, keepdim=True)  # average pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # max pooling
        
        # merge the two pooled maps
        spatial_attention = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        
        # apply the spatial attention
        return x_channel * spatial_attention


class RGBOnlyContrastModule(nn.Module):
    """
    RGB-only contrast module.

    Multi-scale context contrast that replaces the original RGB-D contrast
    branch.

    Arguments:
    planes: input feature dimension.
    """
    def __init__(self, planes):
        super(RGBOnlyContrastModule, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 8)  # each branch outputs 1/8 of the input channels

        # local features from a small receptive field
        self.local = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        
        # multi-scale context features
        self.context_1 = nn.Sequential(  # medium receptive field
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_2 = nn.Sequential(  # larger receptive field
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_3 = nn.Sequential(  # largest receptive field
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())

        # fuse the six contrast features back to the input width
        self.fusion = nn.Sequential(
            nn.Conv2d(self.outplanes * 6, self.inplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU()
        )

        # attention over the fused contrast features
        self.SELayer = SELayer(self.inplanes)

    def forward(self, x):
        """
        Forward pass.

        Arguments:
        x: input feature map [B, C, H, W].

        Returns:
        contrast features [B, C, H, W] (same channel count as input).
        """
        # extract features at several scales
        local = self.local(x)        # local features
        context_1 = self.context_1(x)  # medium context
        context_2 = self.context_2(x)  # wide context
        context_3 = self.context_3(x)  # very wide context

        # pairwise contrast (difference) between scales
        ccl_01 = local - context_1    # local vs medium
        ccl_02 = local - context_2    # local vs wide
        ccl_03 = local - context_3    # local vs very wide
        ccl_12 = context_1 - context_2  # medium vs wide
        ccl_13 = context_1 - context_3  # medium vs very wide
        ccl_23 = context_2 - context_3  # wide vs very wide

        # concatenate all the contrast features
        contrast_features = torch.cat((ccl_01, ccl_02, ccl_03, ccl_12, ccl_13, ccl_23), 1)
        
        # project the contrast features back to the input width
        fused_features = self.fusion(contrast_features)
        
        # attention refinement
        output = self.SELayer(fused_features)
        
        return output


class RGBOnlyGlassNet(nn.Module):
    """
    RGB-only glass detection network.

    A glass segmentation model optimised for RGB-only input, enhanced for
    low-light and weakly reflective scenes.

    Highlights:
    1. No depth branch; fully adapted to RGB-only input.
    2. An illumination-invariant module for low-light robustness.
    3. A reflection enhancement module for faint glass reflections.
    4. Enhanced attention for better feature selection.
    5. A multi-scale decoder for more accurate masks.

    Arguments:
    backbone_path: path to a pretrained ResNeXt101 checkpoint; None
    falls back to random initialisation.
    """
    def __init__(self, backbone_path=None):
        super(RGBOnlyGlassNet, self).__init__()
        
        # load the ResNeXt101 backbone
        resnext = ResNeXt101(backbone_path)
        self.layer0 = resnext.layer0  # input conv layer
        self.layer1 = resnext.layer1  # stage-1 features
        self.layer2 = resnext.layer2  # stage-2 features
        self.layer3 = resnext.layer3  # stage-3 features
        self.layer4 = resnext.layer4  # stage-4 features

        # illumination-invariant features, stable under harsh lighting
        # note: these modules take the reduced features (outputs of cr4, cr3, cr2, cr1)
        self.illumination_invariant_4 = IlluminationInvariantModule(512, 512)  # level 4: 512->512
        self.illumination_invariant_3 = IlluminationInvariantModule(256, 256)  # level 3: 256->256
        self.illumination_invariant_2 = IlluminationInvariantModule(128, 128)  # level 2: 128->128
        self.illumination_invariant_1 = IlluminationInvariantModule(64, 64)    # level 1: 64->64

        # reflection enhancement for weakly reflective glass
        self.reflection_enhancer_4 = ReflectionEnhancementModule(512)  # level 4
        self.reflection_enhancer_3 = ReflectionEnhancementModule(256)  # level 3
        self.reflection_enhancer_2 = ReflectionEnhancementModule(128)  # level 2
        self.reflection_enhancer_1 = ReflectionEnhancementModule(64)   # level 1

        # attention module to sharpen feature selection
        self.enhanced_attention_4 = EnhancedAttentionModule(512)  # level 4
        self.enhanced_attention_3 = EnhancedAttentionModule(256)  # level 3
        self.enhanced_attention_2 = EnhancedAttentionModule(128)  # level 2
        self.enhanced_attention_1 = EnhancedAttentionModule(64)   # level 1

        # RGB-only contrast module (replaces the RGB-D version)
        self.rgb_contrast_4 = RGBOnlyContrastModule(512)  # level 4
        self.rgb_contrast_3 = RGBOnlyContrastModule(256)  # level 3
        self.rgb_contrast_2 = RGBOnlyContrastModule(128)  # level 2
        self.rgb_contrast_1 = RGBOnlyContrastModule(64)   # level 1

        # channel reduction down to a common width
        self.cr4 = nn.Sequential(nn.Conv2d(2048, 512, 1, 1, 0), nn.BatchNorm2d(512), nn.ReLU())  # 2048->512
        self.cr3 = nn.Sequential(nn.Conv2d(1024, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU())  # 1024->256
        self.cr2 = nn.Sequential(nn.Conv2d(512, 128, 1, 1, 0), nn.BatchNorm2d(128), nn.ReLU())   # 512->128
        self.cr1 = nn.Sequential(nn.Conv2d(256, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU())     # 256->64

        # upsampling stages that recover spatial resolution
        self.up_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 2x upsampling
            nn.Conv2d(512, 256, 3, 1, 1),                  # 512->256
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.up_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 2x upsampling
            nn.Conv2d(256, 128, 3, 1, 1),                  # 256->128
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.up_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 2x upsampling
            nn.Conv2d(128, 64, 3, 1, 1),                   # 128->64
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.up_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 2x upsampling
            nn.Conv2d(64, 32, 3, 1, 1),                    # 64->32
            nn.BatchNorm2d(32),
            nn.ReLU())

        # prediction head that outputs the final mask
        self.layer4_predict = nn.Conv2d(256, 1, 3, 1, 1)  # level-4 prediction
        self.layer3_predict = nn.Conv2d(128, 1, 3, 1, 1)  # level-3 prediction
        self.layer2_predict = nn.Conv2d(64, 1, 3, 1, 1)   # level-2 prediction
        self.layer1_predict = nn.Conv2d(32, 1, 3, 1, 1)   # level-1 prediction

        # activation
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True  # in-place ReLU to save memory

    def forward(self, x):
        """
        Forward pass.

        Arguments:
        x: input RGB image [B, 3, H, W].

        Returns:
        layer1_predict: finest prediction [B, 1, H, W].
        layer2_predict: [B, 1, H, W].
        layer3_predict: [B, 1, H, W].
        layer4_predict: coarsest prediction [B, 1, H, W].
        """
        # 1. backbone features
        layer0 = self.layer0(x)   # input features [B, 64, H/2, W/2]
        layer1 = self.layer1(layer0)  # stage 1 [B, 256, H/4, W/4]
        layer2 = self.layer2(layer1)  # stage 2 [B, 512, H/8, W/8]
        layer3 = self.layer3(layer2)  # stage 3 [B, 1024, H/16, W/16]
        layer4 = self.layer4(layer3)  # stage 4 [B, 2048, H/32, W/32]

        # 2. channel reduction to a common width
        rgb_layer4 = self.cr4(layer4)  # 2048->512
        rgb_layer3 = self.cr3(layer3)  # 1024->256
        rgb_layer2 = self.cr2(layer2)  # 512->128
        rgb_layer1 = self.cr1(layer1)  # 256->64

        # 3. illumination-invariant features
        illum4 = self.illumination_invariant_4(rgb_layer4)  # level-4 illumination-invariant features
        illum3 = self.illumination_invariant_3(rgb_layer3)  # level-3 illumination-invariant features
        illum2 = self.illumination_invariant_2(rgb_layer2)  # level-2 illumination-invariant features
        illum1 = self.illumination_invariant_1(rgb_layer1)  # level-1 illumination-invariant features

        # 4. reflection enhancement
        reflect4 = self.reflection_enhancer_4(illum4)  # level-4 reflection-enhanced features
        reflect3 = self.reflection_enhancer_3(illum3)  # level-3 reflection-enhanced features
        reflect2 = self.reflection_enhancer_2(illum2)  # level-2 reflection-enhanced features
        reflect1 = self.reflection_enhancer_1(illum1)  # level-1 reflection-enhanced features

        # 5. attention for feature selection
        attn4 = self.enhanced_attention_4(reflect4)  # level-4 attention outputs
        attn3 = self.enhanced_attention_3(reflect3)  # level-3 attention outputs
        attn2 = self.enhanced_attention_2(reflect2)  # level-2 attention outputs
        attn1 = self.enhanced_attention_1(reflect1)  # level-1 attention outputs

        # 6. multi-scale contrast features
        contrast4 = self.rgb_contrast_4(attn4)  # level-4 contrast features
        contrast3 = self.rgb_contrast_3(attn3)  # level-3 contrast features
        contrast2 = self.rgb_contrast_2(attn2)  # level-2 contrast features
        contrast1 = self.rgb_contrast_1(attn1)  # level-1 contrast features

        # 7. multi-scale decoder with progressive fusion
        up4 = self.up_4(contrast4)  # level-4 upsampling
        layer4_predict = self.layer4_predict(up4)  # level-4 prediction

        up3 = self.up_3(contrast3 + up4)  # level-3 upsampling, fused with level-4 features
        layer3_predict = self.layer3_predict(up3)  # level-3 prediction

        up2 = self.up_2(contrast2 + up3)  # level-2 upsampling, fused with level-3 features
        layer2_predict = self.layer2_predict(up2)  # level-2 prediction

        up1 = self.up_1(contrast1 + up2)  # level-1 upsampling, fused with level-2 features
        layer1_predict = self.layer1_predict(up1)  # level-1 prediction

        # 8. upsample back to the input resolution
        layer4_predict = F.upsample(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.upsample(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.upsample(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.upsample(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        return layer1_predict, layer2_predict, layer3_predict, layer4_predict


def detect_environment():
    """
    Detect the current runtime environment.

    Returns:
    dict: environment details.
    """
    import platform
    import sys
    
    env_info = {
        'system': platform.system(),
        'machine': platform.machine(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__ if 'torch' in sys.modules else 'Not loaded',
        'cuda_available': torch.cuda.is_available() if 'torch' in sys.modules else False,
    }
    
    # check whether we are on an Apple Silicon Mac
    if env_info['system'] == 'Darwin' and 'arm' in env_info['machine'].lower():
        env_info['is_mac_m_series'] = True
    else:
        env_info['is_mac_m_series'] = False
    
    # detect the CUDA version
    if env_info['cuda_available']:
        env_info['cuda_version'] = torch.version.cuda
    else:
        env_info['cuda_version'] = None
    
    return env_info


def test_model_without_pretrained():
    """
    Smoke-test the model with random weights.

    Only checks the structure; no external weight file is required.
    """
    print("开始测试模型（不使用预训练权重）...")
    
    # detect the environment
    env = detect_environment()
    print("环境检测结果:")
    for key, value in env.items():
        print(f"  {key}: {value}")
    
    # dummy input for a quick check
    img = np.random.rand(2, 3, 384, 384).astype(np.float32)
    img = torch.from_numpy(img)
    
    # random-init model, no pretrained weights
    print("\n创建模型实例（随机初始化）...")
    net = RGBOnlyGlassNet(backbone_path=None)
    
    # run a forward pass
    print("执行前向传播...")
    output = net(img)
    
    # print the output shapes
    print("\nRGB-only玻璃检测模型测试结果:")
    print("=" * 50)
    print(f"输入形状: {img.shape}")
    print(f"输出数量: {len(output)}")
    for i, out in enumerate(output):
        print(f"输出 {i+1} 形状: {out.shape}")
    print("=" * 50)
    print("模型测试完成！")
    
    return net, output


def test_model_with_pretrained_if_available():
    """
    Smoke-test the model, using the pretrained weights when available.

    Falls back to random initialisation if the weights cannot be loaded.
    """
    import os
    
    print("开始测试模型（尝试使用预训练权重）...")
    
    # detect the environment
    env = detect_environment()
    print("环境检测结果:")
    for key, value in env.items():
        print(f"  {key}: {value}")
    
    # check whether the pretrained weights exist
    pretrained_path = 'resnext_101_32x4d.pth'
    if os.path.exists(pretrained_path):
        print(f"\n找到预训练权重文件: {pretrained_path}")
        print("使用预训练权重初始化模型...")
        backbone_path = pretrained_path
    else:
        print(f"\n未找到预训练权重文件: {pretrained_path}")
        print("使用随机初始化...")
        backbone_path = None
    
    # dummy input for a quick check
    img = np.random.rand(2, 3, 384, 384).astype(np.float32)
    img = torch.from_numpy(img)
    
    # create the model
    net = RGBOnlyGlassNet(backbone_path=backbone_path)
    
    # run a forward pass
    output = net(img)
    
    # print the output shapes
    print("\nRGB-only玻璃检测模型测试结果:")
    print("=" * 50)
    print(f"输入形状: {img.shape}")
    print(f"输出数量: {len(output)}")
    for i, out in enumerate(output):
        print(f"输出 {i+1} 形状: {out.shape}")
    print("=" * 50)
    print("模型测试完成！")
    
    return net, output


if __name__ == '__main__':
    """
    模型测试主函数
    提供多种测试选项
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='RGB-only玻璃检测模型测试')
    parser.add_argument('--mode', type=str, default='auto', 
                       choices=['auto', 'pretrained', 'random'],
                       help='测试模式: auto(自动检测), pretrained(使用预训练权重), random(随机初始化)')
    
    args = parser.parse_args()
    
    print("RGB-only玻璃检测模型测试")
    print("=" * 60)
    
    if args.mode == 'auto':
        test_model_with_pretrained_if_available()
    elif args.mode == 'pretrained':
        # prefer the pretrained weights when available
        import os
        pretrained_path = 'resnext_101_32x4d.pth'
        if os.path.exists(pretrained_path):
            print(f"使用预训练权重: {pretrained_path}")
            img = np.random.rand(2, 3, 384, 384).astype(np.float32)
            img = torch.from_numpy(img)
            net = RGBOnlyGlassNet(backbone_path=pretrained_path)
            output = net(img)
            print(f"模型测试完成！输出形状: {[out.shape for out in output]}")
        else:
            print(f"错误: 未找到预训练权重文件 {pretrained_path}")
            print("请使用 --mode random 进行测试")
    elif args.mode == 'random':
        test_model_without_pretrained()
    
    print("=" * 60)
    print("测试完成！")
