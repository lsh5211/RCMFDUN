import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class Residual_Block(nn.Module):
    def __init__(self, input_channel):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)

    def forward(self, x_input):
        out = self.conv2(self.relu(self.conv1(x_input)))
        return out + x_input


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class MAPF_Module(nn.Module):
    def __init__(self, input_channel):
        super(MAPF_Module, self).__init__()

        self.Conv3x3_1 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)
        self.RCAB_1 = RCAB(default_conv, input_channel, kernel_size=3, reduction=2)

        self.Down_1 = nn.Conv2d(input_channel, input_channel * 2, kernel_size=3, stride=2, padding=1)
        self.RCAB_2 = RCAB(default_conv, input_channel * 2, kernel_size=3, reduction=2)

        self.Down_2 = nn.Conv2d(input_channel * 2, input_channel * 4, kernel_size=3, stride=2, padding=1)
        self.RCAB_3 = RCAB(default_conv, input_channel * 4, kernel_size=3, reduction=2)

        self.Up_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Conv3x3_2 = nn.Conv2d(input_channel * 6, input_channel * 2, kernel_size=3, padding=1)
        self.RCAB_4 = RCAB(default_conv, input_channel * 2, kernel_size=3, reduction=2)

        self.Up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Conv3x3_3 = nn.Conv2d(input_channel * 3, input_channel, kernel_size=3, padding=1)
        self.RCAB_5 = RCAB(default_conv, input_channel, kernel_size=3, reduction=2)

    def forward(self, x_input):
        x1 = self.Conv3x3_1(x_input)
        x1 = self.RCAB_1(x1)

        x2 = self.Down_1(x1)
        x2 = self.RCAB_2(x2)

        x3 = self.Down_2(x2)
        x3 = self.RCAB_3(x3)

        x3_up = self.Up_1(x3)
        x4 = torch.cat([x2, x3_up], dim=1)
        x4 = self.Conv3x3_2(x4)
        x4 = self.RCAB_4(x4)

        x4_up = self.Up_2(x4)
        x5 = torch.cat([x1, x4_up], dim=1)
        x5 = self.Conv3x3_3(x5)
        x_out = self.RCAB_5(x5)

        return x_out


class FRF_Module(nn.Module):
    def __init__(self, input_channel):
        super(FRF_Module, self).__init__()

        self.Convolution_3_1 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)
        self.Convolution_3_2 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)
        self.Relu = nn.ReLU()

        self.rcab = RCAB(default_conv, input_channel, kernel_size=3, reduction=2)

    def forward(self, x_input, x_e):
        x_e = self.Convolution_3_1(x_e)
        x_e = self.Relu(x_e)
        x_e = self.Convolution_3_2(x_e)

        x = x_input + x_e
        x = self.rcab(x)

        return x


def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0, stride=32, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    temp = nn.PixelShuffle(32)(temp)
    return temp


class Basic_Block(nn.Module):
    def __init__(self, dim):
        super(Basic_Block, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([0.5]))
        self.conv1 = nn.Conv2d(1, dim, kernel_size=3, padding=1)  # ← 新增
        self.FPN = MAPF_Module(dim)
        self.IF = FRF_Module(dim)
        self.cat = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)

    def forward(self, x, r, z, Phi, PhiT):
        r = self.alpha * r

        x = self.conv1(x)
        x = self.cat(torch.cat((x, z), dim=1))
        x_feat = self.FPN(x)
        x_feat = self.IF(x, x_feat)
        z = x_feat

        x_single = self.conv2(x_feat)
        n = x_single - self.alpha * PhiTPhi_fun(x_single, Phi, PhiT)
        x_out = r + n
        return x_out, z



class Net(nn.Module):
    def __init__(self, sensing_rate, LayerNo, channel_number):
        super(Net, self).__init__()
        self.LayerNo = LayerNo
        self.measurement = int(sensing_rate * 1024)  # 32×32=1024
        self.base = channel_number

        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.measurement, 1024)))

        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.base, 1, kernel_size=1, padding=0, bias=True)

        self.Residual_Block = Residual_Block(self.base)

        self.alpha = nn.Parameter(torch.Tensor([1.0]))

        onelayer = []
        for _ in range(self.LayerNo):
            onelayer.append(Basic_Block(self.base))
        self.RND = nn.ModuleList(onelayer)

    def forward(self, x):

        Phi = self.Phi.contiguous().view(self.measurement, 1, 32, 32)
        PhiT = self.Phi.t().contiguous().view(1024, self.measurement, 1, 1)

        y = F.conv2d(x, Phi, padding=0, stride=32, bias=None)

        x_rec = F.conv2d(y, PhiT, padding=0, bias=None)
        x_rec = nn.PixelShuffle(32)(x_rec)

        r = self.alpha * x_rec

        x = self.conv1(x_rec)
        x_feat = self.Residual_Block(x)

        z = x_feat

        x_single = self.conv2(x_feat)

        n = x_single - self.alpha * PhiTPhi_fun(x_single, Phi, PhiT)
        x_out = r + n

        for i in range(self.LayerNo):
            x_out, z = self.RND[i](x_out, r, z, Phi, PhiT)

        return x_out
