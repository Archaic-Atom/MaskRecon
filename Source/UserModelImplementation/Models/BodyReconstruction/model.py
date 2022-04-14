import torch
import torch.nn as nn
import functools
#import sys
#sys.path.append("./Source/UserModelImplementation/Models/BodyReconstruction")
from .submodel import UNet, SUNet

class ColorModel(nn.Module):
    """docstring for ColorModel"""

    def __init__(self, in_channel: int, ngf: int) -> object:
        super().__init__()
        
        #self.color_net = submodel.UNet(in_channel, 3, ngf)
        self.color_net = UNet(in_channel, out_channel=3, ngf=ngf, upconv=False, norm=True)
    
    
    def forward(self, color_img: torch.tensor, depth_img: torch.tensor, uv_img: torch.tensor) -> torch.tensor:
        #color_front = self.color_net(torch.cat((color_img, depth_img), dim=1),
        #                         inter_mode='bilinear')
        color_front = self.color_net(torch.cat((color_img, depth_img, uv_img), dim=1))
        #print(depth_front.shape)

        return color_front

class DepthModel(nn.Module):
    """docstring for DepthMode"""

    def __init__(self, in_channel: int, ngf: int) -> object:
        super().__init__()
        
        #self.color_net = submodel.UNet(in_channel, 3, ngf)
        self.depth_net = SUNet(in_channel, out_channel=1, ngf=ngf, upconv=False, norm=True)
        
    
    def forward(self, color_img: torch.tensor, depth_img: torch.tensor, uv_img: torch.tensor) -> torch.tensor:
        #color_front = self.color_net(torch.cat((color_img, depth_img), dim=1),
        #                         inter_mode='bilinear')
        depth_front = self.depth_net(torch.cat((color_img, depth_img, uv_img), dim=1))
        #print(depth_front.shape)

        return depth_front

class GeneratorModel(nn.Module):
    """docstring for DepthMode"""

    def __init__(self, in_channel: int, ngf: int) -> object:
        super().__init__()
        
        #self.color_net = submodel.UNet(in_channel, 3, ngf)
        self.color_net = UNet(in_channel, out_channel=3, ngf=ngf, upconv=False, norm=True)
        self.depth_net = SUNet(in_channel, out_channel=1, ngf=ngf, upconv=False, norm=True)
        
    
    def forward(self, color_img: torch.tensor, depth_img: torch.tensor, uv_img: torch.tensor) -> torch.tensor:
        #color_front = self.color_net(torch.cat((color_img, depth_img), dim=1),
        #                         inter_mode='bilinear')
        color_front = self.color_net(torch.cat((color_img, depth_img, uv_img), dim=1))
        depth_front = self.depth_net(torch.cat((color_img, depth_img, uv_img), dim=1))
        #print(depth_front.shape)

        return color_front, depth_front

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