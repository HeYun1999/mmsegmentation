# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList
from ..utils import resize
from ...datasets.transforms.my_datapreprocess import generate_edge_main

class Resize(BaseModule):
    def __init__(self,
                 trans_in_channels = [32,64,160,256],
                 resnet_in_channels=[256, 512, 1024, 2048],
                 out_channel = 256,
                 shape = 128,
                 align_corners=False,
                 ):
        super().__init__()

        self.trans_in_channels = trans_in_channels
        self.resnet_in_channels = resnet_in_channels
        self.out_channel = out_channel
        self.shape = shape
        self.align_corners = align_corners

        self.rechannels = nn.ModuleList([])
        in_channels = self.trans_in_channels + self.resnet_in_channels
        for in_channel in in_channels :
            self.rechannels.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, self.out_channel, 1, 1, bias=False),
                    nn.BatchNorm2d(self.out_channel),
                    nn.ReLU(),
                    nn.Conv2d(self.out_channel, self.out_channel, 1, 1, bias=False),
                    nn.BatchNorm2d(self.out_channel),
                    nn.ReLU()
                )
            )
    def forward(self, inputs:list):
        outs = []
        for i,input in enumerate(inputs):
            input = self.rechannels[i](input)
            input = resize(
                input=input,
                size=self.shape,
                mode='bilinear',
                align_corners=self.align_corners)
            outs.append(input)
        return outs

class decoupling(BaseModule):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 ):
        super().__init__()

        self.SGE = SpatialGroupEnhance(8)
        self.convout_mod = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels , channels, 3, 2, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Upsample(scale_factor=2)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, channels, 3, 2, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Upsample(scale_factor=2)
            )
            ])

    def forward(self,inputs):
        trans_inputs = inputs[:4]
        resnet_inputs = inputs[4:]
        decoupled = []
        for i in range(len(inputs)//2):
            sub = trans_inputs[i] - resnet_inputs[i]
            decoupled.append(sub)
        transed = torch.cat(trans_inputs,dim=1)
        decoupled = torch.cat(decoupled,dim=1)
        transed = self.convout_mod[0](transed)
        decoupled = self.convout_mod[1](decoupled)

        global_feature = self.SGE(transed)
        decoupled_feature = self.SGE(decoupled)
        main_feature = global_feature + decoupled_feature
        outs = [global_feature,decoupled_feature,main_feature]


        return outs


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





@MODELS.register_module()
class TransResnetHead(BaseDecodeHead):

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 align_corners=False,
                 norm_cfg = dict(type='BN', requires_grad=True),
                 ignore_index = 255,
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            **kwargs)
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.ignore_index = ignore_index

        self.resize = Resize(shape=self.in_channels)
        self.decoupling  = decoupling(in_channels=256*4,channels=256*2)

        self.fusion_conv = ConvModule(
            in_channels=512,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)


    def forward(
            self,
            inputs:list):
        """Forward function.
        Args:
            inputs (Tensor | tuple[Tensor]): Input tensor or tuple of
                Tensor. When training, the input is a tuple of three tensors,
                (p_feat, i_feat, d_feat), and the output is a tuple of three
                tensors, (p_seg_logit, i_seg_logit, d_seg_logit).
                When inference, only the head of integral branch is used, and
                input is a tensor of integral feature map, and the output is
                the segmentation logit.

        Returns:
            Tensor | tuple[Tensor]: Output tensor or tuple of tensors.
        """
        x = self.resize(inputs)
        global_feature, decoupled_feature, main_feature = self.decoupling(x)
        if self.training:
            global_feature = self.fusion_conv(global_feature)
            decoupled_feature = self.fusion_conv(decoupled_feature)
            main_feature = self.fusion_conv(main_feature)

            global_feature = self.cls_seg(global_feature)
            decoupled_feature = self.cls_seg(decoupled_feature)
            main_feature = self.cls_seg(main_feature)

            return global_feature, decoupled_feature, main_feature
        else:
            main_feature = self.fusion_conv(main_feature)
            main_feature = self.cls_seg(main_feature)
            return main_feature

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tuple[Tensor]:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        gt_edge_segs = []
        gt_main_segs = []
        for gt_semantic_seg in gt_semantic_segs:
            gt_egde,gt_main = generate_edge_main(gt_semantic_seg)
            gt_edge_segs.append(gt_egde)
            gt_main_segs.append(gt_main)

        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)#torch.stack新建一个dim = 0维度，并在这个维度上拼接
        gt_main_segs = torch.stack(gt_main_segs, dim=0)
        return gt_sem_segs, gt_edge_segs,gt_main_segs

    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        global_feature, decoupled_feature, main_feature = seg_logits
        sem_label, edge_label,main_label = self._stack_batch_gt(batch_data_samples)
        global_feature = resize(
            input=global_feature,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        decoupled_feature = resize(
            input=decoupled_feature,
            size=edge_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        main_feature = resize(
            input=main_feature,
            size=main_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        sem_label = sem_label.squeeze(1)
        edge_label = edge_label.squeeze(1)
        main_label = main_label.squeeze(1)

        loss['loss_sem'] = self.loss_decode[0](global_feature, sem_label)
        loss['loss_edge'] = self.loss_decode[1](decoupled_feature, edge_label)
        loss['loss_main'] = self.loss_decode[2](main_feature, main_label)

        loss['acc_seg'] = accuracy(
            main_feature, main_label, ignore_index=self.ignore_index)
        return loss
