# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)

from mmseg.registry import MODELS
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw,resize
from .resnet import ResNet,ResNetV1c
from .mit import MixFFN,EfficientMultiheadAttention,TransformerEncoderLayer
@MODELS.register_module()
class ResNet_i(ResNetV1c):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        pretrained= 'open-mmlab://resnet18_v1c'
        self.pretrained = pretrained
    def forward(self,x,index):
        """Forward function."""
        layer_n = ['layer1','layer2','layer3','layer4']
        if index == 0:
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)
        outs = []
        #layer_name:layer1,layer2,layer3,layer4
        res_layer = getattr(self, layer_n[index])
        x = res_layer(x)
        return x
@MODELS.register_module()
class fusion_decoupe(BaseModule):
    def __init__(self):
        super().__init__()

        self.resize_trans = nn.ModuleList([
            nn.Sequential(),
            nn.Sequential(),
            nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2)
        ])
        self.decoupe_resnet = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(96, 64, 1, 1, 0),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(192, 128, 1, 1, 0),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(416, 256, 1, 1, 0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        ])

        self.decoupe_trans = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(96, 32, 1, 1, 0),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(192, 64, 1, 1, 0),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(416, 160, 1, 1, 0),
                nn.BatchNorm2d(160),
                nn.ReLU(),

            )
        ])
        self.ECAAttention = ECAAttention()


    def forward(self,resnet_input,trans_input,i):
        trans_input = self.resize_trans[i](trans_input)
        cat = torch.cat((resnet_input,trans_input),dim=1)
        #cat_maxpooled = nn.functional.max_pool2d(cat, 2)

        cat = self.ECAAttention(cat)
        if i !=3:
            resnet_output = self.decoupe_resnet[i](cat)
            trans_output = self.decoupe_trans[i](cat)
        else:
            resnet_output = 0
            trans_output = 0

        return resnet_output,trans_output,cat

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

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
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)
@MODELS.register_module()
class ConvFormerNet(BaseModule):
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False,

                 #resnet_init
                 resnet_name='ResNetV1c',
                 resnet_depth=50,
                 resnet_num_stages=4,
                 resnet_out_indices=(0, 1, 2, 3),
                 resnet_dilations=(1, 1, 2, 4),
                 resnet_strides=(1, 2, 1, 1),
                 resnet_norm_cfg= dict(type='LN', eps=1e-6),
                 resnet_norm_eval=False,
                 resnet_style='pytorch',
                 resnet_contract_dilation=True
                 ):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))#每一个
            cur += num_layer

            #Resnet
        self.resnet_name = resnet_name,
        self.resnet_i = ResNet_i(
            depth = resnet_depth,
            num_stages = resnet_num_stages,
            out_indices = resnet_out_indices,
            dilations = resnet_dilations,
            strides = resnet_strides,
            norm_cfg = resnet_norm_cfg,
            norm_eval = resnet_norm_eval,
            style = resnet_style,
            contract_dilation = resnet_contract_dilation
        )
        #resnet与transformer融合模块
        self.fusion_decoupe =fusion_decoupe()
    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        outs = []
        #resnet out put :64,128,128  128,64,64  256,64,64  512,64,64
        for i, layer in enumerate(self.layers):#layers保存了四个transformer块
            if i == 0:
                resnet_x = self.resnet_i(x, index=i)
            else:
                resnet_x = self.resnet_i(resnet_x, index=i)
            x, hw_shape = layer[0](x)#layer[0]是PatchEmbed，嵌入模块
            for block in layer[1]:#layer[1]是循环两次的TransformerEncoderLayer 模块
                x = block(x, hw_shape)
            x = layer[2](x)#layer[2]是一个LayerNorm，标准化模块
            x = nlc_to_nchw(x, hw_shape)#由于transformer块出来的不是nchw的维度形式，而是nlc的维度，因此将他重整为nchw
            #transformer output :32,128,128  64,64,64  160,32,32   256,64,64
            resnet_x,x,cat = self.fusion_decoupe(resnet_x,x,i)
            outs.append(cat)
        return outs
