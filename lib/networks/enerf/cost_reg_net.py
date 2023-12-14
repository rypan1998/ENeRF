import torch.nn as nn
from .utils import *

class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=nn.BatchNorm3d):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))
        self.depth_conv = nn.Sequential(nn.Conv3d(8, 1, 3, padding=1, bias=False))
        self.feat_conv = nn.Sequential(nn.Conv3d(8, 8, 3, padding=1, bias=False))

    def forward(self, x):
        conv0 = self.conv0(x) # 8*H*W
        conv2 = self.conv2(self.conv1(conv0)) # 16*H/2*W/2
        conv4 = self.conv4(self.conv3(conv2)) # 32*H/4*W/4
        x = self.conv6(self.conv5(conv4)) # 64*H/8*W/8
        x = conv4 + self.conv7(x) # 32*H/4*W/4
        del conv4
        x = conv2 + self.conv9(x) # 16*H/2*W/2
        del conv2
        x = conv0 + self.conv11(x) # 8*H*W
        del conv0
        feat = self.feat_conv(x) # 8*H*W
        depth = self.depth_conv(x) # H*W
        return feat, depth.squeeze(1)


class MinCostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=nn.BatchNorm3d):
        super(MinCostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))

        self.depth_conv = nn.Sequential(nn.Conv3d(8, 1, 3, padding=1, bias=False))
        self.feat_conv = nn.Sequential(nn.Conv3d(8, 8, 3, padding=1, bias=False))

    def forward(self, x):
        # coarse 阶段 x = 32，代表输入通道数 C
        # D 代表“深度”，指数据在三维空间的一个维度。比如 (x,y,z) 中 z 就是指深度
        conv0 = self.conv0(x) # B*32*D*H*W -> B*8*D*H*W
        conv2 = self.conv2(self.conv1(conv0)) # B*16*D*H/2*W/2
        conv4 = self.conv4(self.conv3(conv2)) # B*32*D*H/4*W/4
        x = conv4
        x = conv2 + self.conv9(x) # B*16*D*H/2*W/2
        del conv2
        x = conv0 + self.conv11(x) # B*8*D*H*W
        del conv0
        feat = self.feat_conv(x) # B*8*D*H*W
        depth = self.depth_conv(x) # B*1*D*H*W
        return feat, depth.squeeze(1) # 消除第 1 维的大小为 1 的元素（不为 1 则不消除）
