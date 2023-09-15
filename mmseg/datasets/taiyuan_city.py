# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class TaiyuanCityDataset(BaseSegDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('farmland', 'woodland', 'grassland', 'waters', 'building',
               'Hardened_surface', 'Heap_digging', 'road','others','background'),
        palette=[[0, 255, 0], [34, 139, 34], [107, 142, 35], [0, 0, 255],
                         [255, 0, 0], [192, 192, 192],[128, 42, 42],[254, 252, 193],[255, 255, 255],[250,250,250]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            ignore_index= 9,#使得图片中标签为9的像素值全部变为255
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
