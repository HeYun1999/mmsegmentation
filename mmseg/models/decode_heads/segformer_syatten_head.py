# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class FR(nn.Module):
    def __init__(self,c=3,h=128,w=128):
        super().__init__()

        self.pooling_h0= nn.AvgPool2d(kernel_size=(128, 1), stride=1, padding=0)
        self.pooling_h1 = nn.AvgPool2d(kernel_size=(64, 1), stride=1, padding=0)
        self.pooling_h2 = nn.AvgPool2d(kernel_size=(67, 1), stride=1, padding=0)


        self.pooling_w0 = nn.AvgPool2d(kernel_size=(1, 128), stride=1, padding=0)
        self.pooling_w1 = nn.AvgPool2d(kernel_size=(1, 64), stride=1, padding=0)
        self.pooling_w2 = nn.AvgPool2d(kernel_size=(1, 67), stride=1, padding=0)

    def forward(self, x):
        pooling_h0 = self.pooling_h0(x)
        pooling_h1 = self.pooling_h1(x)
        pooling_h2 = self.pooling_h2(x)
        pooling_h = [pooling_h0,pooling_h1,pooling_h2]
        pooling_h = torch.cat(pooling_h, dim=2)

        pooling_w0 = self.pooling_w0(x)
        pooling_w1 = self.pooling_w1(x)
        pooling_w2 = self.pooling_w2(x)
        pooling_w = [pooling_w0,pooling_w1,pooling_w2]
        pooling_w = torch.cat(pooling_w, dim=3)

        pooling = [pooling_h,pooling_w]
        pooling = torch.cat(pooling, dim=3)#(8,1024,128,256)
        pooling = torch.transpose(pooling, 1, 2)#(8,256,1024,128)
        return pooling
class Sy_Attention_Model(nn.Module):
    def  __init__(self):
        super().__init__()
        self.fr = FR()
        self.conv = nn.Conv2d(1024, 1024, (1,7),(1,2),(0,3))

    def forward(self,x):
        feature0=self.fr(x)#(8,256,1024,128)(b,h+w,c,s)
        feature1 = torch.transpose(feature0, 2, 3)
        feature_mul = torch.matmul(feature0,feature1)#(8,128,1024,1024)即A（i，j）
        feature_mul = torch.matmul(feature_mul,feature0)#(8,128,1024,256)
        feature_ed = feature_mul + feature0 #(8,128,1024,256)
        feature_ed = torch.transpose(feature_ed, 1, 2)
        feature_ed = self.conv(feature_ed)
        #feature_ed = torch.reshape(feature_ed,(8,1024,128,128))



        return feature_ed
class Segformer_Syatten_Head(BaseDecodeHead):
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

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.Sy_Attention_Model = Sy_Attention_Model()

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        out = torch.cat(outs, dim=1)
        #协同注意力↓

        out = self.Sy_Attention_Model(out)
        #协同注意力↑
        out = self.fusion_conv(out)

        out = self.cls_seg(out)

        return out
