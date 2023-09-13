# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.my_decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
import math
import torch.nn.functional as F

@MODELS.register_module()
class Segformer_Decoupling_Head(BaseDecodeHead):
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
            in_channels=768,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.fusion_conv2 = ConvModule(
            in_channels=512,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        #self.Sy_Attention_Model = Sy_Attention_Model()
        #self.ECALayer = ECALayer(chanel = 1024)
        self.SELayer = SELayer(1024)
        self.decoupling = decoupling(in_channels=2048,out_channels=512)
        #self.CBAM = CBAM(1024)
        self.CoTAttention = CoTAttention(dim=256, kernel_size=3)
        self.PyramidPooling = pyramidPooling(1024,[6, 3, 2, 1])
    def forward(self, inputs):
        #只选取前四个特征图
        or_input = inputs[4]
        or_input = self.CoTAttention(or_input)
        inputs = inputs[:4]
        #第五个为低纬度特征
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)#inputs:
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            out = resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners)
            out = self.CoTAttention(out)
            outs.append(out)
        out = torch.cat(outs, dim=1)
        #低高维度结合↓
        out = self.PyramidPooling(out)
        out = self.decoupling(out)#输出两个张量，out[0]为类别5的特征，out[1]为剩下的特征
        out_decoupling = out[0]
        out_main = out[0] + out[1]
        outs = [out_main,or_input]
        out_cir = torch.cat(outs, dim=1)
        #out = self.ECALayer(out)
        #out = self.Sy_Attention_Model(out)
        #低高维度结合↑
        out_main = self.fusion_conv1(out_cir)
        out_main = self.cls_seg(out_main)

        out_decoupling = self.fusion_conv2(out_decoupling)
        out_decoupling = self.cls_seg(out_decoupling)

        outs = [out_decoupling,out_main]

        return outs

class decoupling(nn.Module):
    def __init__(self,in_channels=2048,out_channels=1024,c=3,h=128,w=128):
        super().__init__()
        self.conv1_mod = nn.Sequential(
        nn.Conv2d(in_channels, in_channels//4, 3, 2,padding=1, bias=False),
        nn.BatchNorm2d(in_channels//4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels//4, in_channels//4, 3, 2, padding=1,bias=False),
        nn.BatchNorm2d(in_channels//4),
        nn.Conv2d(in_channels//4, in_channels, 1, 1, bias=False),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=4)
        )
        self.conv2_mod = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
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


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        self.cbr_unit = nn.Sequential(conv_mod,
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes):
        super(pyramidPooling, self).__init__()

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=False))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        output_slices = [x]
        h, w = x.shape[2:]

        for module, pool_size in zip(self.path_module_list, self.pool_sizes):
            out = F.avg_pool2d(x, int(h/pool_size), int(h/pool_size), 0)
            out = module(out)
            out = F.upsample(out, size=(h,w), mode='bilinear')
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)


class CoTAttention(nn.Module):

    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)


        return k1+k2
