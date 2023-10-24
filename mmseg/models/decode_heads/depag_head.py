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
from .decoupling_fusion import PagFM
class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


@MODELS.register_module()
class DePagHead(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super().__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
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
        self.pag = PagFM(140,140)
        self.fusion = Fusion(140,140)


    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        #out_put : (6,140,128,128)
        output = self.pag(output)
        decoupled = self.fusion(output)
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

class Fusion(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()

        self.decoupling = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 2, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.Upsample(scale_factor=2),
                nn.ReLU()),

            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 2, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.Upsample(scale_factor=2),
                nn.ReLU()),

            nn.Sequential(
                nn.Conv2d(420, out_channels*2, 3, 2, 1),
                nn.BatchNorm2d(out_channels*2),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(out_channels*2, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.Upsample(scale_factor=2),
                nn.ReLU())
        ])
    def forward(self,edge_main:list) -> list:
        edge_main_all = []
        if isinstance(edge_main, list):
            for i in range(len(edge_main)):
                ou = self.decoupling[i](edge_main[i])
                edge_main_all.append(ou)
            ou = edge_main_all[0] + edge_main_all[1]
            ou = nn.functional.sigmoid(ou)
            ou = torch.cat((ou, edge_main_all[0]), dim=1)
            ou = torch.cat((ou, edge_main_all[1]), dim=1)
            ou = self.decoupling[2](ou)
            edge_main_all.append(ou)
        return edge_main_all