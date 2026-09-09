import torch
from torch import nn

from backbone.resnext import resnext_101_32x4d_


class ResNeXt101(nn.Module):
    def __init__(self, backbone_path):
        super(ResNeXt101, self).__init__()
        net = resnext_101_32x4d_.resnext_101_32x4d
        if backbone_path is not None:
            try:
                weights = torch.load(backbone_path, weights_only=False)
                net.load_state_dict(weights, strict=True)
                print("Load ResNeXt Weights Succeed!")
            except Exception as e:
                print(f"⚠ 加载ResNeXt预训练权重失败: {e}")
                print(f"  路径: {backbone_path}")
                print("  已回退为随机初始化（不影响训练，仅无预训练加速）")

        net = list(net.children())
        '''
        self.freeze_bn = True
        if self.freeze_bn:
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        '''            
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]
        print("Load ResNeXt Weights Finished!")

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4