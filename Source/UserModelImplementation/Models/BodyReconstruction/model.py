import torch
import torch.nn as nn
import functools
from torchvision import transforms
#import sys
#sys.path.append("./Source/UserModelImplementation/Models/BodyReconstruction")
from .submodel import UNet, SUNet, RecoverySizeX4, FeatureFusion
from .extractor import BasicEncoder
import time 
from torchstat import stat

class ColorModel(nn.Module):
    """docstring for ColorModel"""

    def __init__(self, ngf: int, mask: bool) -> object:
        super().__init__()
        self.mask = mask
        self.dropout = 0

        self.color_extarction = BasicEncoder(input_dim=3, output_dim=256, norm_fn='instance', dropout=self.dropout)
        self.recovery_color = RecoverySizeX4(out_channel=3)
        self.condition_extraction = BasicEncoder(input_dim=4, output_dim=256, norm_fn='instance', dropout=self.dropout)
        self.feature_fusion = FeatureFusion(dim=256)
        self.color_net = UNet(out_channel=3, ngf=ngf, upconv=False, norm=True)
        
    
    
    def forward(self, color_img: torch.tensor, depth_img: torch.tensor, uv_img: torch.tensor) -> torch.tensor:
        color_feature = self.color_extarction(color_img, name = "color_feature")

        condition_feature = self.condition_extraction(torch.cat((depth_img, uv_img), dim=1), name = "color_condition_feature")       
        all_feature = self.feature_fusion(color_feature, condition_feature)
        color_front = self.color_net(all_feature)
        if self.training and self.mask:
            recovery_color = self.recovery_color(color_feature)
            return [color_front, recovery_color]
        return [color_front]

class DepthModel(nn.Module):
    """docstring for ColorModel"""

    def __init__(self, ngf: int, mask: bool) -> object:
        super().__init__()
        self.mask = mask
        self.dropout = 0

        self.depth_extarction = BasicEncoder(input_dim=1, output_dim=256, norm_fn='instance', dropout=self.dropout)
        self.recovery_depth = RecoverySizeX4(out_channel=1)
        self.condition_extraction = BasicEncoder(input_dim=6, output_dim=256, norm_fn='instance', dropout=self.dropout)
        self.feature_fusion = FeatureFusion(dim=256)
        self.depth_net = SUNet(out_channel=1, ngf=ngf, upconv=False, norm=True)
        
    
    
    def forward(self, color_img: torch.tensor, depth_img: torch.tensor, uv_img: torch.tensor) -> torch.tensor:
        depth_feature = self.depth_extarction(depth_img, name = "depth_feature")
        condition_feature = self.condition_extraction(torch.cat((color_img, uv_img), dim=1), name = "depth_condition_feature")       
        all_feature = self.feature_fusion(depth_feature, condition_feature)
        depth_front = self.depth_net(all_feature)
        # depth_front = self.depth_net(depth_feature)
        if self.training and self.mask:
            recovery_depth = self.recovery_depth(depth_feature)
            return [depth_front, recovery_depth]
        return [depth_front]

            

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)