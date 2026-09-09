"""
Ablation variants of the full model, built by adding one module at a
time on top of a plain baseline.

Pipeline: rgb_layer -> illumination -> reflection -> attention -> contrast

Variants:
  baseline:          rgb_layer -> contrast
  plus_illumination: rgb_layer -> illumination -> contrast
  plus_reflection:   rgb_layer -> illumination -> reflection -> contrast
  plus_attention:    rgb_layer -> illumination -> reflection -> attention -> contrast (= full)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_rgb_only import RGBOnlyGlassNet


class RGBOnlyGlassNet_Baseline(RGBOnlyGlassNet):
    """
    Baseline without any of the three core modules; only the backbone,
    contrast, upsampling and prediction head remain.
    """
    def __init__(self, backbone_path=None):
        super(RGBOnlyGlassNet_Baseline, self).__init__(backbone_path)
        # All three modules are replaced with identity mappings.
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
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        rgb_layer4 = self.cr4(layer4)
        rgb_layer3 = self.cr3(layer3)
        rgb_layer2 = self.cr2(layer2)
        rgb_layer1 = self.cr1(layer1)

        # All three modules are skipped.
        illum4 = self.illumination_invariant_4(rgb_layer4)
        illum3 = self.illumination_invariant_3(rgb_layer3)
        illum2 = self.illumination_invariant_2(rgb_layer2)
        illum1 = self.illumination_invariant_1(rgb_layer1)

        reflect4 = self.reflection_enhancer_4(illum4)
        reflect3 = self.reflection_enhancer_3(illum3)
        reflect2 = self.reflection_enhancer_2(illum2)
        reflect1 = self.reflection_enhancer_1(illum1)

        attn4 = self.enhanced_attention_4(reflect4)
        attn3 = self.enhanced_attention_3(reflect3)
        attn2 = self.enhanced_attention_2(reflect2)
        attn1 = self.enhanced_attention_1(reflect1)

        contrast4 = self.rgb_contrast_4(attn4)
        contrast3 = self.rgb_contrast_3(attn3)
        contrast2 = self.rgb_contrast_2(attn2)
        contrast1 = self.rgb_contrast_1(attn1)

        up4 = self.up_4(contrast4)
        layer4_predict = self.layer4_predict(up4)
        up3 = self.up_3(contrast3 + up4)
        layer3_predict = self.layer3_predict(up3)
        up2 = self.up_2(contrast2 + up3)
        layer2_predict = self.layer2_predict(up2)
        up1 = self.up_1(contrast1 + up2)
        layer1_predict = self.layer1_predict(up1)

        layer4_predict = F.interpolate(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.interpolate(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.interpolate(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.interpolate(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        return layer1_predict, layer2_predict, layer3_predict, layer4_predict


class RGBOnlyGlassNet_PlusIllumination(RGBOnlyGlassNet):
    """
    Baseline plus the illumination invariant module.
    Reflection enhancement and enhanced attention are skipped.
    """
    def __init__(self, backbone_path=None):
        super(RGBOnlyGlassNet_PlusIllumination, self).__init__(backbone_path)
        # Only illumination is kept; reflection and attention are skipped.
        self.reflection_enhancer_4 = nn.Identity()
        self.reflection_enhancer_3 = nn.Identity()
        self.reflection_enhancer_2 = nn.Identity()
        self.reflection_enhancer_1 = nn.Identity()
        self.enhanced_attention_4 = nn.Identity()
        self.enhanced_attention_3 = nn.Identity()
        self.enhanced_attention_2 = nn.Identity()
        self.enhanced_attention_1 = nn.Identity()

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        rgb_layer4 = self.cr4(layer4)
        rgb_layer3 = self.cr3(layer3)
        rgb_layer2 = self.cr2(layer2)
        rgb_layer1 = self.cr1(layer1)

        # Illumination invariant module is active.
        illum4 = self.illumination_invariant_4(rgb_layer4)
        illum3 = self.illumination_invariant_3(rgb_layer3)
        illum2 = self.illumination_invariant_2(rgb_layer2)
        illum1 = self.illumination_invariant_1(rgb_layer1)

        # Reflection enhancement and attention are skipped.
        reflect4 = self.reflection_enhancer_4(illum4)
        reflect3 = self.reflection_enhancer_3(illum3)
        reflect2 = self.reflection_enhancer_2(illum2)
        reflect1 = self.reflection_enhancer_1(illum1)

        attn4 = self.enhanced_attention_4(reflect4)
        attn3 = self.enhanced_attention_3(reflect3)
        attn2 = self.enhanced_attention_2(reflect2)
        attn1 = self.enhanced_attention_1(reflect1)

        contrast4 = self.rgb_contrast_4(attn4)
        contrast3 = self.rgb_contrast_3(attn3)
        contrast2 = self.rgb_contrast_2(attn2)
        contrast1 = self.rgb_contrast_1(attn1)

        up4 = self.up_4(contrast4)
        layer4_predict = self.layer4_predict(up4)
        up3 = self.up_3(contrast3 + up4)
        layer3_predict = self.layer3_predict(up3)
        up2 = self.up_2(contrast2 + up3)
        layer2_predict = self.layer2_predict(up2)
        up1 = self.up_1(contrast1 + up2)
        layer1_predict = self.layer1_predict(up1)

        layer4_predict = F.interpolate(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.interpolate(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.interpolate(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.interpolate(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        return layer1_predict, layer2_predict, layer3_predict, layer4_predict


class RGBOnlyGlassNet_PlusReflection(RGBOnlyGlassNet):
    """
    Baseline plus illumination invariant and reflection enhancement.
    Enhanced attention is skipped.
    """
    def __init__(self, backbone_path=None):
        super(RGBOnlyGlassNet_PlusReflection, self).__init__(backbone_path)
        # Illumination and reflection are active; only attention is skipped.
        self.enhanced_attention_4 = nn.Identity()
        self.enhanced_attention_3 = nn.Identity()
        self.enhanced_attention_2 = nn.Identity()
        self.enhanced_attention_1 = nn.Identity()

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        rgb_layer4 = self.cr4(layer4)
        rgb_layer3 = self.cr3(layer3)
        rgb_layer2 = self.cr2(layer2)
        rgb_layer1 = self.cr1(layer1)

        # Illumination invariant module is active.
        illum4 = self.illumination_invariant_4(rgb_layer4)
        illum3 = self.illumination_invariant_3(rgb_layer3)
        illum2 = self.illumination_invariant_2(rgb_layer2)
        illum1 = self.illumination_invariant_1(rgb_layer1)

        # Reflection enhancement module is active.
        reflect4 = self.reflection_enhancer_4(illum4)
        reflect3 = self.reflection_enhancer_3(illum3)
        reflect2 = self.reflection_enhancer_2(illum2)
        reflect1 = self.reflection_enhancer_1(illum1)

        # Attention is skipped.
        attn4 = self.enhanced_attention_4(reflect4)
        attn3 = self.enhanced_attention_3(reflect3)
        attn2 = self.enhanced_attention_2(reflect2)
        attn1 = self.enhanced_attention_1(reflect1)

        contrast4 = self.rgb_contrast_4(attn4)
        contrast3 = self.rgb_contrast_3(attn3)
        contrast2 = self.rgb_contrast_2(attn2)
        contrast1 = self.rgb_contrast_1(attn1)

        up4 = self.up_4(contrast4)
        layer4_predict = self.layer4_predict(up4)
        up3 = self.up_3(contrast3 + up4)
        layer3_predict = self.layer3_predict(up3)
        up2 = self.up_2(contrast2 + up3)
        layer2_predict = self.layer2_predict(up2)
        up1 = self.up_1(contrast1 + up2)
        layer1_predict = self.layer1_predict(up1)

        layer4_predict = F.interpolate(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.interpolate(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.interpolate(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.interpolate(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        return layer1_predict, layer2_predict, layer3_predict, layer4_predict


class RGBOnlyGlassNet_PlusAttention(RGBOnlyGlassNet):
    """
    Full model: all three modules enabled. Reuses RGBOnlyGlassNet.forward.
    """
    pass  # All module behavior comes from the parent class, i.e. the full model.


# Registry of ablation variants, in the order modules are added.
MODEL_VARIANTS = {
    'baseline':          RGBOnlyGlassNet_Baseline,
    'plus_illumination': RGBOnlyGlassNet_PlusIllumination,
    'plus_reflection':   RGBOnlyGlassNet_PlusReflection,
    'plus_attention':    RGBOnlyGlassNet_PlusAttention,
}