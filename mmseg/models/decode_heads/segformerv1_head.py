# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_headv1 import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
from torch.nn import init

@MODELS.register_module()
class SegformerHeadV1(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv_decoupling = ConvModule(
            in_channels=256,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.fusion_conv_main = ConvModule(
            in_channels=320,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.SGE = SpatialGroupEnhance(8)


        self.decoupling = decoupling(1024,256)
    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        or_inputs =[]
        for idx in range(len(inputs)):
            x = inputs[idx]
            or_inputs.append(x)
            #x = self.SGE(x)
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        out = torch.cat(outs, dim=1)
        out = self.decoupling(out)
        decoupling = out[0]
        main = out[1]
        outs = [or_inputs[0],main]
        main = torch.cat(outs, dim=1)

        decoupling = self.fusion_conv_decoupling(decoupling)
        decoupling = self.cls_seg(decoupling)

        main = self.fusion_conv_main(main)
        main = self.cls_seg(main)

        outs = [decoupling,main]
        return outs


class decoupling(nn.Module):
    def __init__(self,in_channels=2304,out_channels=1024):
        super().__init__()
        self.decoupling_mod = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2,padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Upsample(scale_factor=2)
            )
        self.convout_mod = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels , out_channels, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            ])

    def forward(self, x):
        out = self.decoupling_mod(x)
        out_ed = x - out
        outs=[out,out_ed]
        out_conv = []
        for idex in range(len(outs)):
            ou = self.convout_mod[idex](outs[idex])
            out_conv.append(ou)
        out_add = out_conv[0] + out_conv[1]

        outs = [out_conv[0],out_add]
        return outs#2,256,256,256





class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.contiguous().view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        return x