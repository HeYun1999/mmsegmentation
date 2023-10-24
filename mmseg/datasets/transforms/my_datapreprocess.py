# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS

import torch
import torch.nn as nn
import random
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS

@TRANSFORMS.register_module()
class MyFlip(BaseTransform):
    def __init__(self, direction: str):
        super().__init__()
        self.direction = direction

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = mmcv.imflip(img, direction=self.direction)
        return results
@TRANSFORMS.register_module()
class GenerateEdgeMain(BaseTransform):
    def __init__(self, width: int = 3,ignore_index: int = 255) -> None:
        super().__init__()
        self.width = width
        self.ignore_index = ignore_index

    def transform(self, results: Dict) -> Dict:
        """Call function to generate edge from segmentation map.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with edge mask.
        """
        h, w = results['img_shape']
        edge = np.zeros((h, w), dtype=np.uint8)#产生同等大小的一个数组
        seg_map = results['gt_seg_map']#seg_map是标签图

        # down     边缘特征图的下部分处理
        edge_down = edge[1:h, :]#
        edge_down[(seg_map[1:h, :] != seg_map[:h - 1, :])
                  & (seg_map[1:h, :] != self.ignore_index) &
                  (seg_map[:h - 1, :] != self.ignore_index)] = 1
        # left
        edge_left = edge[:, :w - 1]
        edge_left[(seg_map[:, :w - 1] != seg_map[:, 1:w])
                  & (seg_map[:, :w - 1] != self.ignore_index) &
                  (seg_map[:, 1:w] != self.ignore_index)] = 1
        # up_left
        edge_upleft = edge[:h - 1, :w - 1]
        edge_upleft[(seg_map[:h - 1, :w - 1] != seg_map[1:h, 1:w])
                    & (seg_map[:h - 1, :w - 1] != self.ignore_index) &
                    (seg_map[1:h, 1:w] != self.ignore_index)] = 1
        # up_right
        edge_upright = edge[:h - 1, 1:w]
        edge_upright[(seg_map[:h - 1, 1:w] != seg_map[1:h, :w - 1])
                     & (seg_map[:h - 1, 1:w] != self.ignore_index) &
                     (seg_map[1:h, :w - 1] != self.ignore_index)] = 1

        main = np.where(edge==1,0,1)
        results['gt_main_map'] = main
        results['width'] = self.width


        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'width={self.width}, '
        repr_str += f'ignore_index={self.ignore_index})'

        return repr_str

def generate_edge_main(original_labels):
    # 预处理标签，将相邻像素值相同的位置设置为-1
    processed_labels = original_labels.clone()  # 创建预处理后的标签副本
    processed_labels = processed_labels.unsqueeze(1).cpu()
    original_labels_clone = processed_labels
    # 使用卷积操作检查相邻像素
    conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    conv.weight.data = torch.tensor([[[[1, 1, 1], [1, -8, 1], [1, 1, 1]]]], dtype=torch.float32)
    # 将标签转换为张量
    labels_tensor = processed_labels.float()
    # 应用卷积操作
    processed_labels = conv(labels_tensor)

    # 将相同的位置设置为-1
    label_edge = torch.where(processed_labels == 0, 255, original_labels_clone)
    label_main = torch.where(processed_labels != 0, 255, original_labels_clone)
    label_edge = label_edge.squeeze(1).to(torch.int64).cuda()
    label_main = label_main.squeeze(1).to(torch.int64).cuda()
    return label_edge,label_main