# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.my_decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
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
        #num_inputs = len(self.in_channels)
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

        self.fusion_conv1 = ConvModule(
            in_channels=self.channels * (num_inputs + 1),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.fusion_conv2 = ConvModule(
            in_channels=self.channels * (num_inputs),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.Sy_Attention_Model = Sy_Attention_Model()
        self.ECALayer = ECALayer()
        self.SELayer = SELayer(1024)
        self.decoupling = decoupling()
    def forward(self, inputs):
        #只选取前四个特征图
        or_input = inputs[4]
        inputs = inputs[:4]
        #第五个为低纬度特征
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)#inputs:
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
        #低高维度结合↓
        out = self.SELayer(out)
        out = self.decoupling(out)
        out2 = out[0]
        out = out[0] + out[1]
        outs = [out,or_input]
        out = torch.cat(outs, dim=1)
        #out = self.ECALayer(out)
        #out = self.Sy_Attention_Model(out)
        #低高维度结合↑
        out1 = self.fusion_conv1(out)
        out1 = self.cls_seg(out1)

        out2 = self.fusion_conv2(out2)
        out2 = self.cls_seg(out2)

        outs = [out1,out2]

        return outs

class decoupling(nn.Module):
    def __init__(self,c=3,h=128,w=128):
        super().__init__()
        self.conv1_mod = nn.Sequential(
        nn.Conv2d(1024, 512, 3, 2,padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, 2, padding=1,bias=False),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 1024, 1, 1, bias=False),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=4)
        )
        self.conv2_mod = nn.Sequential(
        nn.Conv2d(1024, 1024, 3, 1, padding=1, bias=False),
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace=True)
        )

    def forward(self, x):

        out = self.conv1_mod(x)
        out_ed = x - out
        outs=[out,out_ed]
        out_conv = []
        for ou in outs:
            ou = self.conv2_mod(ou)
            out_conv.append(ou)
        return out_conv

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
#SE注意力机制
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return out

#eca注意力机制
class ECALayer(nn.Module):
    def __init__(self, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1280, 1280, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y) ## y为每个通道的权重值
        out = x * y.expand_as(x)  ##将y的通道权重一一赋值给x的对应通道
        return out
