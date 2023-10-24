# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from ..utils import resize
from .aspp_head import ASPPHead, ASPPModule
import torch.nn.functional as F
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
from .decoupling_fusion import PagFM,Fusion



@MODELS.register_module()
class DecouplingFusionHead(BaseDecodeHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self,c1_channels,**kwargs):
        super().__init__(**kwargs)
        self.input_transform = 'resize_concat'
        self.c1_channels =c1_channels

        self.conv = nn.Sequential(
                nn.Conv2d(1472, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
        )
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.pag = PagFM(256,256)
        self.fusion = Fusion(256,128)

        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                128,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def transform_inputs(self, inputs):
        upsampled_inputs = [
            resize(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for x in inputs
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)
        inputs = self.conv(inputs)


        return inputs


    def forward(self, inputs):
        """Forward function."""
        x = self.transform_inputs(inputs)
        output = self.pag(x)
        decoupled = self.fusion(x,output)
        out = []
        if isinstance(decoupled,list):
            for i in decoupled:
                ou = self.sep_bottleneck(i)
                ou = self.cls_seg(ou)
                out.append(ou)
        if not self.training:
            out = out[2]

        return out


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

        gt_all_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)#torch.stack新建一个dim = 0维度，并在这个维度上拼接
        gt_main_segs = torch.stack(gt_main_segs, dim=0)
        return  gt_edge_segs,gt_main_segs,gt_all_segs

    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        edge_feature, main_feature, all_feature = seg_logits
        edge_label,main_label,all_label = self._stack_batch_gt(batch_data_samples)
        edge_feature = resize(
            input=edge_feature,
            size=edge_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        main_feature = resize(
            input=main_feature,
            size=main_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        all_feature = resize(
            input=all_feature,
            size=all_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)


        edge_label = edge_label.squeeze(1)
        main_label = main_label.squeeze(1)
        all_label = all_label.squeeze(1)

        loss['loss_main'] = self.loss_decode[0](main_feature, main_label)
        loss['loss_edge'] = self.loss_decode[1](edge_feature, edge_label)
        loss['loss_all'] = self.loss_decode[2](all_feature, all_label,ignore_index = 255)

        loss['acc_seg'] = accuracy(
            all_feature, all_label, ignore_index=self.ignore_index)
        return loss

