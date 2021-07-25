""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # base = 48
        # base = 36  # ?
        # base = 32  # ?
        # base = 24  # ok
        # base = 16  # not good
        base = 20

        self.inc = DoubleConv(n_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base * 8, base * 16 // factor)
        self.up1 = Up(base * 16, base * 8 // factor, bilinear)
        self.up2 = Up(base * 8, base * 4 // factor, bilinear)
        self.up3 = Up(base * 4, base * 2 // factor, bilinear)
        self.up4 = Up(base * 2, base, bilinear)
        self.outc = OutConv(base, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
