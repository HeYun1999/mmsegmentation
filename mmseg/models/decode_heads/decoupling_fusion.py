import torch.nn.functional as F
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType

class PagFM(BaseModule):
    """Pixel-attention-guided fusion module.

    Args:
        in_channels (int): The number of input channels.
        channels (int): The number of channels.
        after_relu (bool): Whether to use ReLU before attention.
            Default: False.
        with_channel (bool): Whether to use channel attention.
            Default: False.
        upsample_mode (str): The mode of upsample. Default: 'bilinear'.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(typ='ReLU', inplace=True).
        init_cfg (dict): Config dict for initialization. Default: None.
    """
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 after_relu: bool = False,
                 with_channel: bool = True,
                 upsample_mode: str = 'bilinear',
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(typ='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.after_relu = after_relu
        self.with_channel = with_channel
        self.upsample_mode = upsample_mode

        self.f_i = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.f_p = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)

        if with_channel:
            self.up = ConvModule(
                channels, in_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

        if after_relu:
            self.relu = MODELS.build(act_cfg)

        self.changeconv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,2,1),
            nn.BatchNorm2d(in_channels),
            #nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x_p: Tensor) -> Tensor:
        """Forward function.

        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.

        Returns:
            Tensor: The feature map with pixel-attention-guided fusion.
        """
        x_i = self.changeconv(x_p)
        if self.after_relu:
            x_p = self.relu(x_p)
            x_i = self.relu(x_i)

        f_i = self.f_i(x_i)
        f_i = F.interpolate(
            f_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)

        f_p = self.f_p(x_p)

        if self.with_channel:
            sigma = torch.sigmoid(self.up(f_p * f_i))
        else:
            sigma = torch.sigmoid(torch.sum(f_p * f_i, dim=1).unsqueeze(1))

        mask = torch.zeros_like(x_p)

        out_high = torch.where(sigma > 0.5,x_p,mask)

        out_low = torch.where(sigma <= 0.5,x_p,mask)
        #out = sigma * x_i + (1 - sigma) * x_p
        edge_main = [out_low,out_high]
        return edge_main

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
                nn.Conv2d(out_channels*2, out_channels*2, 3, 2, 1),
                nn.BatchNorm2d(out_channels*2),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(out_channels*2, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.Upsample(scale_factor=2),
                nn.ReLU())
        ])
    def forward(self,x,edge_main:list) -> list:
        edge_main_all = []
        if isinstance(edge_main, list):
            for i in range(len(edge_main)):
                ou = self.decoupling[i](edge_main[i])
                edge_main_all.append(ou)

            '''
            ou = edge_main_all[0] + edge_main_all[1]
            ou = nn.functional.sigmoid(ou)
            ou = torch.cat((ou, edge_main_all[0]), dim=1)
            ou = torch.cat((ou, edge_main_all[1]), dim=1)
            ou = torch.cat((ou, x), dim=1)
            '''
            ou = torch.cat((edge_main_all[0],edge_main_all[1]),dim=1)
            ou = self.decoupling[2](ou)
            edge_main_all.append(ou)
        return edge_main_all
