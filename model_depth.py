import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from backbone.resnext.resnext101_regular import ResNeXt101


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4, num_context=8):
        super(SELayer, self).__init__()
        self.channel = channel
        self.num_context = num_context
        self.context_channel = int(channel / num_context)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.context_attention = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, num_context, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, 1, 0, groups=num_context, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, 1, 0, groups=num_context, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        context_attention = self.context_attention(y)
        channel_attention = self.channel_attention(y)
        context_attention = context_attention.repeat(1, 1, self.context_channel, 1)
        context_attention = context_attention.view(-1, self.channel, 1, 1)
        attention = context_attention * channel_attention
        return x * attention.expand_as(x)

class DenseContrastModule(nn.Module):
    def __init__(self, planes):
        super(DenseContrastModule, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 8)

        self.local_1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_2 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_3 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())

        self.SELayer = SELayer(int(self.inplanes / 8 * 6))

    def forward(self, x):
        local_1 = self.local_1(x)
        context_1 = self.context_1(x)
        ccl_01 = local_1 - context_1

        context_2 = self.context_2(x)
        ccl_02 = local_1 - context_2

        context_3 = self.context_3(x)
        ccl_03 = local_1 - context_3

        ccl_12 = context_1 - context_2
        ccl_13 = context_1 - context_3
        ccl_23 = context_2 - context_3

        output = torch.cat((ccl_01, ccl_02, ccl_03, ccl_12, ccl_13, ccl_23), 1)
        output = self.SELayer(output)
        return output

class RGBDContrastModule(nn.Module):
    def __init__(self, planes):
        super(RGBDContrastModule, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 8)

        self.rgb_local = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_2 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_3 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        
        self.depth_local = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.depth_context_1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.depth_context_2 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.depth_context_3 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        
        self.fused_local = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.fused_context_1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.fused_context_2 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.fused_context_3 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())

        self.SELayer = SELayer(int(self.inplanes / 8 * 18))

    def forward(self, rgb, depth, fused):
        rgb_local = self.rgb_local(rgb)
        rgb_context_1 = self.context_1(rgb)
        rgb_ccl_01 = rgb_local - rgb_context_1

        rgb_context_2 = self.context_2(rgb)
        rgb_ccl_02 = rgb_local - rgb_context_2

        rgb_context_3 = self.context_3(rgb)
        rgb_ccl_03 = rgb_local - rgb_context_3

        rgb_ccl_12 = rgb_context_1 - rgb_context_2
        rgb_ccl_13 = rgb_context_1 - rgb_context_3
        rgb_ccl_23 = rgb_context_2 - rgb_context_3



        depth_local = self.depth_local(depth)
        depth_context_1 = self.depth_context_1(depth)
        depth_ccl_01 = depth_local - depth_context_1

        depth_context_2 = self.depth_context_2(depth)
        depth_ccl_02 = depth_local - depth_context_2

        depth_context_3 = self.depth_context_3(depth)
        depth_ccl_03 = depth_local - depth_context_3

        depth_ccl_12 = depth_context_1 - depth_context_2
        depth_ccl_13 = depth_context_1 - depth_context_3
        depth_ccl_23 = depth_context_2 - depth_context_3

        fused_local = self.fused_local(fused)
        fused_context_1 = self.fused_context_1(fused)
        fused_ccl_01 = fused_local - fused_context_1

        fused_context_2 = self.fused_context_2(fused)
        fused_ccl_02 = fused_local - fused_context_2

        fused_context_3 = self.fused_context_3(fused)
        fused_ccl_03 = fused_local - fused_context_3

        fused_ccl_12 = fused_context_1 - fused_context_2
        fused_ccl_13 = fused_context_1 - fused_context_3
        fused_ccl_23 = fused_context_2 - fused_context_3

        output = torch.cat(
            (
                rgb_ccl_01, rgb_ccl_02, rgb_ccl_03, rgb_ccl_12, rgb_ccl_13, rgb_ccl_23,
                depth_ccl_01, depth_ccl_02, depth_ccl_03, depth_ccl_12, depth_ccl_13, depth_ccl_23,
                fused_ccl_01, fused_ccl_02, fused_ccl_03, fused_ccl_12, fused_ccl_13, fused_ccl_23, 
            ), 1)
        output = self.SELayer(output)
        return output

class RGBD_Asymmetric_Module(nn.Module):
    def __init__(self, planes):
        super(RGBD_Asymmetric_Module, self).__init__()
        self.inplanes = int(planes)

        self.outplanes = int(planes / 8)

        self.depth_inplanes = int(planes / 4)
        self.depth_outplanes = int(planes / 16)

        self.rgb_local = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_2 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_3 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        
        self.depth_local = nn.Sequential(
            nn.Conv2d(self.depth_inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.depth_context_1 = nn.Sequential(
            nn.Conv2d(self.depth_inplanes, self.outplanes, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.depth_context_2 = nn.Sequential(
            nn.Conv2d(self.depth_inplanes, self.outplanes, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.depth_context_3 = nn.Sequential(
            nn.Conv2d(self.depth_inplanes, self.outplanes, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        
        self.fused_layer = nn.Sequential(
            nn.Conv2d(self.inplanes + self.depth_inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())


        self.fused_local = nn.Sequential(
            nn.Conv2d(self.outplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.fused_context_1 = nn.Sequential(
            nn.Conv2d(self.outplanes, self.outplanes, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.fused_context_2 = nn.Sequential(
            nn.Conv2d(self.outplanes, self.outplanes, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.fused_context_3 = nn.Sequential(
            nn.Conv2d(self.outplanes, self.outplanes, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())

        
        self.cross_local = nn.Sequential(
            nn.Conv2d(self.outplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.cross_context_1 = nn.Sequential(
            nn.Conv2d(self.outplanes, self.outplanes, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.cross_context_2 = nn.Sequential(
            nn.Conv2d(self.outplanes, self.outplanes, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.cross_context_3 = nn.Sequential(
            nn.Conv2d(self.outplanes, self.outplanes, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())

        self.rgb_SELayer = SELayer(int(self.inplanes))
        self.depth_SELayer = SELayer(int(self.inplanes))
        self.fused_SELayer = SELayer(int(self.inplanes))

        self.cross_SELayer = SELayer(int(self.inplanes))

        self.final_fused = nn.Sequential(
            nn.Conv2d(self.inplanes * 4, self.inplanes, kernel_size=1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
        )

        self.final_SELayer = SELayer(int(self.inplanes))



    def forward(self, rgb, depth):
        rgb_local = self.rgb_local(rgb)
        rgb_context_1 = self.context_1(rgb)
        rgb_ccl_01 = rgb_local + rgb_context_1

        rgb_context_2 = self.context_2(rgb)
        rgb_ccl_02 = rgb_local + rgb_context_2

        rgb_context_3 = self.context_3(rgb)
        rgb_ccl_03 = rgb_local + rgb_context_3

        rgb_ccl_12 = rgb_context_1 + rgb_context_2
        rgb_ccl_13 = rgb_context_1 + rgb_context_3
        rgb_ccl_23 = rgb_context_2 + rgb_context_3

        rgb_concat = torch.cat((rgb_local,  rgb_context_1, rgb_ccl_01, rgb_ccl_02, rgb_ccl_03, rgb_ccl_12, rgb_ccl_13, rgb_ccl_23 ), 1) # [c +> inplance]
        rgb_output = self.rgb_SELayer(rgb_concat)

        depth_local = self.depth_local(depth)
        depth_context_1 = self.depth_context_1(depth)
        depth_ccl_01 = depth_local + depth_context_1

        depth_context_2 = self.depth_context_2(depth)
        depth_ccl_02 = depth_local + depth_context_2

        depth_context_3 = self.depth_context_3(depth)
        depth_ccl_03 = depth_local + depth_context_3

        depth_ccl_12 = depth_context_1 + depth_context_2
        depth_ccl_13 = depth_context_1 + depth_context_3
        depth_ccl_23 = depth_context_2 + depth_context_3

        depth_concat = torch.cat(( depth_local, depth_context_1, depth_ccl_01, depth_ccl_02, depth_ccl_03, depth_ccl_12, depth_ccl_13, depth_ccl_23 ), 1) # [c -> inplance]
        depth_output = self.depth_SELayer(depth_concat)


        fused = torch.cat( (rgb, depth), dim=1)

        fused = self.fused_layer(fused)
        fused_local = self.fused_local(fused)
        fused_context_1 = self.fused_context_1(fused)
        fused_ccl_01 = fused_local + fused_context_1

        fused_context_2 = self.fused_context_2(fused)
        fused_ccl_02 = fused_local + fused_context_2

        fused_context_3 = self.fused_context_3(fused)
        fused_ccl_03 = fused_local + fused_context_3

        fused_ccl_12 = fused_context_1 + fused_context_2
        fused_ccl_13 = fused_context_1 + fused_context_3
        fused_ccl_23 = fused_context_2 + fused_context_3

        fused_concat = torch.cat(( fused_local, fused_context_1, fused_ccl_01, fused_ccl_02, fused_ccl_03, fused_ccl_12, fused_ccl_13, fused_ccl_23 ), 1) # [c -> inplance]
        fused_output = self.fused_SELayer(fused_concat)


        cross_feature = rgb_local + depth_local
        cross_local = self.cross_local(cross_feature)
        # cross_context_1 = self.cross_context_1(rgb_ccl_01 + depth_ccl_01)
        cross_ccl_01 = rgb_ccl_01 + depth_ccl_01

        # cross_context_2 = self.cross_context_2(rgb_ccl_02 + depth_ccl_02)
        cross_ccl_02 = rgb_ccl_02 + depth_ccl_02

        # cross_context_3 = self.cross_context_3(rgb_ccl_03 + depth_ccl_03)
        cross_ccl_03 = rgb_ccl_03 + depth_ccl_03

        cross_ccl_12 = self.cross_context_1(rgb_ccl_01 + depth_ccl_02)
        cross_ccl_13 = self.cross_context_2(rgb_ccl_01 + depth_ccl_03)
        cross_ccl_23 = self.cross_context_3(rgb_ccl_01 + depth_ccl_03)

        cross_concat = torch.cat(( cross_local, cross_feature, cross_ccl_01, cross_ccl_02, cross_ccl_03, cross_ccl_12, cross_ccl_13, cross_ccl_23 ), 1) # [c -> inplance]
        cross_output = self.cross_SELayer(cross_concat)


        output = torch.cat( (rgb_output, depth_output, cross_output, fused_output), 1)
        output = self.final_fused(output)
        # output = torch.cat(
        #     (
        #         rgb_ccl_01, rgb_ccl_02, rgb_ccl_03, rgb_ccl_12, rgb_ccl_13, rgb_ccl_23,
        #         depth_ccl_01, depth_ccl_02, depth_ccl_03, depth_ccl_12, depth_ccl_13, depth_ccl_23,
        #         fused_ccl_01, fused_ccl_02, fused_ccl_03, fused_ccl_12, fused_ccl_13, fused_ccl_23, 
        #     ), 1)
        output = self.final_SELayer(output)
        return output, rgb_output, depth_output


class RGBD_NL(nn.Module):
    def __init__(self, rgb_channels, depth_channels, fused_channels):
        super(RGBD_NL, self).__init__()
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.fused_channels = fused_channels

        self.query_rgb = nn.Conv2d(self.rgb_channels, self.rgb_channels // 8, 1, 1, 0)
        self.key_rgb = nn.Conv2d(self.rgb_channels, self.rgb_channels // 8, 1, 1, 0)

        self.query_depth = nn.Conv2d(self.depth_channels, self.depth_channels // 8, 1, 1, 0)
        self.key_depth = nn.Conv2d(self.depth_channels, self.depth_channels // 8, 1, 1, 0)
        self.value_depth = nn.Conv2d(self.depth_channels, self.depth_channels, 1, 1, 0)

        self.query_fused = nn.Conv2d(self.fused_channels, self.fused_channels // 8, 1, 1, 0)
        self.key_fused = nn.Conv2d(self.fused_channels, self.fused_channels // 8, 1, 1, 0)

        self.value = nn.Conv2d(self.fused_channels, self.fused_channels, 1, 1, 0)

        self.gamma_fused = nn.Parameter(torch.zeros(1))
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_depth = nn.Parameter(torch.zeros(1))


        self.softmax_weight = nn.Softmax(dim=-1)


    def forward(self, input_fused, input_rgb, input_depth, input_depthmiss):
        B, C, H, W = input_rgb.size()


        query_fused = self.query_fused(input_fused).view(B, -1, H * W).permute(0, 2, 1)
        key_fused = self.key_fused(input_fused).view(B, -1, H * W)
        energy_fused = torch.bmm(query_fused, key_fused)     
        energy_fused = self.softmax_weight(energy_fused)

        value = self.value(input_fused).view(B, -1, H * W) + input_depthmiss.view(B, -1, H * W)

        # value = self.value(fusion3).view(B, -1, H * W)
        query_rgb = self.query_rgb(input_rgb).view(B, -1, H * W).permute(0, 2, 1)
        key_rgb = self.key_rgb(input_rgb).view(B, -1, H * W)
        energy_rgb = torch.bmm(query_rgb, key_rgb) # [B, H*W, H*W]
        energy_rgb = self.softmax_weight(energy_rgb)
        # # energy_rgb = energy_rgb * weight_rgb_normalized.squeeze(1).expand_as(energy_rgb)


        B_d, C_d, H_d, W_d = input_depth.size()

        query_depth = self.query_depth(input_depth).view(B_d, -1, H_d * W_d).permute(0, 2, 1)
        key_depth = self.key_depth(input_depth).view(B_d, -1, H_d * W_d) 
        energy_depth = torch.bmm(query_depth, key_depth) # [B, H*W, H*W]
        energy_depth = self.softmax_weight(energy_depth)
        value_depth = self.value_depth(input_depth).view(B_d, -1, H_d * W_d) + input_depthmiss.view(B_d, -1, H_d * W_d)

        # energy = energy_rgb + energy_depth
        

        out_final = torch.bmm(value, energy_fused.permute(0, 2, 1))
        out_final = out_final.view(B, C, H, W)
        out_final = self.gamma_fused * out_final + input_fused


        out_rgb = torch.bmm(value, energy_rgb.permute(0, 2, 1)).view(B, C, H, W)
        out_rgb = out_rgb.view(B, C, H, W)
        out_rgb = self.gamma_rgb * out_rgb + input_rgb

        out_depth = torch.bmm(value_depth, energy_depth.permute(0, 2, 1)).view(B_d, C_d, H_d, W_d)
        out_depth = out_depth.view(B_d, C_d, H_d, W_d)        
        out_depth = self.gamma_depth * out_depth + input_depth
   

        # energy_depth = energy_depth * weight_depth_normalized.squeeze(1).expand_as(energy_depth)

        # energy = energy_rgb + energy_depth
        # attention_element = self.softmax_dependency(energy)

        # fusion4 = torch.bmm(value, attention_element.permute(0, 2, 1)).view(B, C, H, W)

        return out_final, out_rgb, out_depth

class RGBD_Fusion(nn.Module):

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(RGBD_Fusion, self).__init__()
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )


    def forward(self, input_rgb: torch.Tensor, input_depth: torch.Tensor) -> torch.Tensor:
        rgbd = torch.cat([input_rgb, input_depth], dim=1)
        feature_pool = self.conv_pool(rgbd)
        return feature_pool



class GlassNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(GlassNet, self).__init__()
        resnext = ResNeXt101(backbone_path)
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        # resnext_depth = ResNeXt101(backbone_path)
        # self.depth_layer0 = resnext_depth.layer0
        # self.depth_layer1 = resnext_depth.layer1
        # self.depth_layer2 = resnext_depth.layer2
        # self.depth_layer3 = resnext_depth.layer3
        # self.depth_layer4 = resnext_depth.layer4
        self.depth_conv0 = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.ReLU(),    
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.depth_conv1 = nn.Sequential(nn.Conv2d(8, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.depth_conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.depth_conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.depth_conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))


        # self.dm_scale_2x = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.ReLU(),    
        #                                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        # self.dm_scale_4x = nn.Sequential(nn.Conv2d(8, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
        #                                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        # self.dm_scale_8x = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
        #                                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        # self.dm_scale_16x = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
        #                                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        # self.dm_scale_32x = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
        #                                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # channel reduction
        self.cr4 = nn.Sequential(nn.Conv2d(2048, 512, 1, 1, 0), nn.BatchNorm2d(512), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(1024, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(512, 128, 1, 1, 0), nn.BatchNorm2d(128), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(256, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU())


        # self.rgbd_fusion_4 = RGBD_Fusion(2048 + 128, 2048)
        # self.rgbd_fusion_3 = RGBD_Fusion(1024 + 64, 1024)
        # self.rgbd_fusion_2 = RGBD_Fusion(512 + 32, 512)
        # self.rgbd_fusion_1 = RGBD_Fusion(256 + 16, 256)

        # self.rgbd_contrast_4 = RGBDContrastModule(2048)
        # self.rgbd_contrast_3 = RGBDContrastModule(1024)
        # self.rgbd_contrast_2 = RGBDContrastModule(512)
        # self.rgbd_contrast_1 = RGBDContrastModule(256)


        # self.rgbd_contrast_4 = DenseContrastModule(2048)
        # self.rgbd_contrast_3 = DenseContrastModule(1024)
        # self.rgbd_contrast_2 = DenseContrastModule(512)
        # self.rgbd_contrast_1 = DenseContrastModule(256)

        self.rgbd_contrast_4 = RGBD_Asymmetric_Module(512)
        self.rgbd_contrast_3 = RGBD_Asymmetric_Module(256)
        self.rgbd_contrast_2 = RGBD_Asymmetric_Module(128)
        self.rgbd_contrast_1 = RGBD_Asymmetric_Module(64)
        
        self.rgbd_NL_4 = RGBD_NL(512, 512, 512)
        self.rgbd_NL_3 = RGBD_NL(256, 256, 256)
        # self.rgbd_NL_2 = RGBD_NL(128, 32, 128)
        # self.rgbd_NL_1 = RGBD_NL(64, 16, 64)

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

        self.rgb_up_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.rgb_up_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.rgb_up_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.rgb_up_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.depth_up_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.depth_up_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.depth_up_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.depth_up_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        # self.up_0 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU())

        self.layer4_predict = nn.Conv2d(256, 1, 3, 1, 1)
        self.layer3_predict = nn.Conv2d(128, 1, 3, 1, 1)
        self.layer2_predict = nn.Conv2d(64, 1, 3, 1, 1)
        self.layer1_predict = nn.Conv2d(32, 1, 3, 1, 1)

        self.depth4_predict = nn.Conv2d(64, 1, 3, 1, 1)
        self.depth3_predict = nn.Conv2d(32, 1, 3, 1, 1)
        self.depth2_predict = nn.Conv2d(16, 1, 3, 1, 1)
        self.depth1_predict = nn.Conv2d(8, 1, 3, 1, 1)


        # self.refine = RefNet(68, 64)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x, depth, depth_missing):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        depth_layer0 = self.depth_conv0(depth)
        depth_layer1 = self.depth_conv1(depth_layer0)
        depth_layer2 = self.depth_conv2(depth_layer1)
        depth_layer3 = self.depth_conv3(depth_layer2)
        depth_layer4 = self.depth_conv4(depth_layer3)

        dm_0 = F.interpolate(depth_missing, scale_factor=0.5)
        dm_1 = F.interpolate(dm_0, scale_factor=0.5)
        dm_2 = F.interpolate(dm_1, scale_factor=0.5)
        dm_3 = F.interpolate(dm_2, scale_factor=0.5)
        dm_4 = F.interpolate(dm_3, scale_factor=0.5)

        # channel reduction
        rgb_layer4 = self.cr4(layer4)
        rgb_layer3 = self.cr3(layer3)
        rgb_layer2 = self.cr2(layer2)
        rgb_layer1 = self.cr1(layer1)

        # rgbd4 = self.rgbd_fusion_4(rgb_layer4, depth_layer4)
        contrast_4, rgb_output4, depth_output4 = self.rgbd_contrast_4(rgb_layer4, depth_layer4)

        
        out_4, rgb_output4, depth_output4 = self.rgbd_NL_4(contrast_4, rgb_output4, depth_output4, dm_4)
        # contrast_4 = self.rgbd_contrast_4(rgbd4)

        up4 = self.up_4(out_4)
        layer4_predict = self.layer4_predict(up4)
        # layer4_map = F.sigmoid(layer4_predict)

        # rgbd3 = self.rgbd_fusion_3(rgb_layer3, depth_layer3)
        rgb_3_shortcut = self.rgb_up_4(rgb_output4)
        depth_3_shortcut = self.depth_up_4(depth_output4)

        depth4_predict = self.depth4_predict(depth_3_shortcut)

        contrast_3, rgb_output3, depth_output3 = self.rgbd_contrast_3(rgb_layer3 + layer4_predict + rgb_3_shortcut, depth_layer3 + depth_3_shortcut) 
        out_3, rgb_output3, depth_output3 = self.rgbd_NL_3(contrast_3, rgb_output3, depth_output3, dm_3)
        up3 = self.up_3(out_3)
        layer3_predict = self.layer3_predict(up3)
        # layer3_map = F.sigmoid(layer3_predict)

        # rgbd2 = self.rgbd_fusion_2(rgb_layer2, depth_layer2)
        rgb_2_shortcut = self.rgb_up_3(rgb_output3)
        depth_2_shortcut = self.depth_up_3(depth_output3)

        depth3_predict = self.depth3_predict(depth_2_shortcut)


        out_2, rgb_output2, depth_output2 = self.rgbd_contrast_2(rgb_layer2 + layer3_predict + rgb_2_shortcut, depth_layer2 + depth_2_shortcut) 
        # out_2, rgb_output2, depth_output2 = self.rgbd_NL_2(out_2, rgb_output2, depth_output2, dm_2)


        # contrast_2 = self.rgbd_contrast_2( rgbd2 * layer3_map)

        up2 = self.up_2(out_2)
        layer2_predict = self.layer2_predict(up2)
        # layer2_map = F.sigmoid(layer2_predict)

        # rgbd1 = self.rgbd_fusion_1(rgb_layer1, depth_layer1)
        rgb_1_shortcut = self.rgb_up_2(rgb_output2)
        depth_1_shortcut = self.depth_up_2(depth_output2)
        depth2_predict = self.depth2_predict(depth_1_shortcut)


        out_1, rgb_output1, depth_output1 = self.rgbd_contrast_1(rgb_layer1 + layer2_predict + rgb_1_shortcut, depth_layer1 + depth_1_shortcut) 
        # out_1, rgb_output1, depth_output1 = self.rgbd_NL_1(out_1, rgb_output1, depth_output1, dm_1)
        # contrast_1 = self.rgbd_contrast_1(rgbd1 * layer2_map)
        up1 = self.up_1(out_1)
        layer1_predict = self.layer1_predict(up1)

        depth_0_shortcut = self.depth_up_1(depth_output1)
        depth1_predict = self.depth1_predict(depth_0_shortcut)

        # layer1_map = F.sigmoid(layer1_predict)
        # layer1_map = F.upsample(layer1_map, size=x.size()[2:], mode='bilinear', align_corners=True)

        # up0 = self.up_0(layer0)
        # layer0_predict, ref = self.refine(torch.cat((x, up0, layer1_map), 1))

        layer4_predict = F.upsample(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.upsample(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.upsample(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.upsample(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)


        depth4_predict = F.upsample(depth4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        depth3_predict = F.upsample(depth3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        depth2_predict = F.upsample(depth2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        depth1_predict = F.upsample(depth1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        # if self.training:
        # return layer0_predict, layer1_predict, layer2_predict, layer3_predict, layer4_predict
        return layer1_predict, layer2_predict, layer3_predict, layer4_predict, \
            depth1_predict, depth2_predict, depth3_predict, depth4_predict


        # return F.sigmoid(layer4_predict), F.sigmoid(layer3_predict), F.sigmoid(layer2_predict), \
        #        F.sigmoid(layer1_predict)


if __name__ == '__main__':
    img = np.random.rand(2, 3, 384, 384).astype(np.float32)
    img = torch.from_numpy(img)
    net = GlassNet(backbone_path='resnext_101_32x4d.pth')
    output = net(img)
