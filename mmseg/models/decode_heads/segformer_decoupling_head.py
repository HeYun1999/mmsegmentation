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
        #低维度特征图
        self.conv_or = nn.Sequential(
            nn.Conv2d(3, 128, 3,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )


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
            in_channels=512,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.decoupling = decoupling(in_channels=2304,out_channels=512)
        self.CoTAttention = nn.ModuleList([
            CoTAttention(dim=64, kernel_size=3),
            CoTAttention(dim=128, kernel_size=3),
            CoTAttention(dim=320, kernel_size=3),
            CoTAttention(dim=512, kernel_size=3),
            CoTAttention(dim=256, kernel_size=3)
            ])

        self.PyramidPooling = pyramidPooling(1024,[6, 3, 2, 1])
        self.fusion_conv1 = ConvModule(
            in_channels=2304,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
    def forward(self, inputs):
        #只选取前四个特征图
        or_input = inputs[4]
        or_input = self.conv_or(or_input)
        or_input = self.CoTAttention[4](or_input)
        inputs = inputs[:4]
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)#inputs:
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            x = self.CoTAttention[idx](x)
            conv = self.convs[idx]
            out = resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners)
            outs.append(out)
        out = torch.cat(outs, dim=1)
        #低高维度结合↓
        out = self.PyramidPooling(out)
        outs =[out,or_input]
        out_cat = torch.cat(outs, dim=1)
        decoupling_value = True
        #decoupling_value = False
        if decoupling_value:
            out = self.decoupling(out_cat)#输出两个张量，out[0]为类别5的特征，out[1]为剩下的特征 均为 2，512，256，256

            out_decoupling = out[0]
            out_main = out[1]

            out_decoupling = self.fusion_conv(out_decoupling)
            out_decoupling = self.cls_seg(out_decoupling)

            out_main = self.fusion_conv(out_main)
            out_main = self.cls_seg(out_main)



            outs = [out_decoupling,out_main]
        else:

            out = self.fusion_conv1(out_cat)
            outs = self.cls_seg(out)

        return outs

class decoupling(nn.Module):
    def __init__(self,in_channels=2304,out_channels=1024,c=3,h=128,w=128):
        super().__init__()
        self.conv0_mod = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            #nn.Upsample(scale_factor=2)#维度降格为四分之一，尺寸变为原来的两倍 756 , 256,256
        )
        self.conv1_mod = nn.Sequential(

            nn.Conv2d(in_channels// 4, in_channels//4, 3, 2,padding=1, bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2)
            )

        self.conv2_mod = nn.Sequential(
            nn.Conv2d(in_channels//4, out_channels, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.conv3_mod = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels // 4, out_channels, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            ])

    def forward(self, x):
        x = self.conv0_mod(x)#2,576,256,256
        #x = x.detach()
        out = self.conv1_mod(x)##尺寸不变，维度不变
        out = torch.mul(x,out)
        out_ed = x - out
        outs=[out,out_ed]
        out_conv = []
        for ou in outs:
            ou = self.conv2_mod(ou)#2,512,256,256
            out_conv.append(ou)
        out_cat = out_conv[0] + out_conv[1]

        outs = [out,out_cat]
        out_conv = []
        for idx in range(len(outs)):
            ou = outs[idx]
            ou = self.conv3_mod[idx](ou)
            out_conv.append(ou)
        return out_conv#2,256,256,256

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
