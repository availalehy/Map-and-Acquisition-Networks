import torch
from torch import nn


class maca_module(nn.Module):
    def __init__(self, input_channel=64):
        super(maca_module, self).__init__()
        self.conv = nn.Conv2d(input_channel, 1, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f = x
        x = self.conv(x)

        # cos distance
        x_l2 = torch.norm(x.expand_as(f).view(x.expand_as(f).size()[0], x.expand_as(f).size()[1], -1), dim=2,
                          keepdim=True)
        f_l2 = torch.norm(f.view(f.size()[0], f.size()[1], -1), dim=2, keepdim=True)
        fm = x_l2 * f_l2
        c = torch.sum((f * x.expand_as(f)).view(f_l2.size()[0], f_l2.size()[1], -1), dim=2, keepdim=True) / fm
        c = torch.unsqueeze(c, dim=-1)
        c = self.sigmoid(c)
        return f * c.expand_as(f)


class masa_module(nn.Module):
    def __init__(self, input_size=[8, 8]):
        super(masa_module, self).__init__()
        self.conv = nn.Conv1d(input_size[0] * input_size[1], 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()

        x = x.view(B, C, -1)
        x = x.permute(0, 2, 1)
        f = x

        x = self.conv(x)

        # cos distance
        x_l2 = torch.norm(x.expand_as(f), dim=-1, keepdim=True)
        f_l2 = torch.norm(f, dim=-1, keepdim=True)
        fm = (x_l2 * f_l2)
        if 0 in fm:
            fm = (x_l2 * f_l2) + 0.00001
        c = torch.sum(f * x.expand_as(f), dim=-1, keepdim=True) / fm
        c = self.sigmoid(c)

        c = f * c.expand_as(f)
        return c.view(B, H, W, C).permute(0, 3, 1, 2)


class masca_moudle(nn.Module):
    def __init__(self, input_channel=64, input_size=[8, 8]):
        super(masca_moudle, self).__init__()
        self.sa = maca_module(input_channel)
        self.ca = masa_module(input_size)
        self.bn = nn.BatchNorm2d(input_channel)

    def forward(self, x):
        out = self.sa(x)
        out = self.bn(out)
        out = self.ca(out)
        return out

class macsa_moudle(nn.Module):
    def __init__(self, input_channel=64, input_size=[8, 8]):
        super(macsa_moudle, self).__init__()
        self.sa = maca_module(input_channel)
        self.ca = masa_module(input_size)
        self.bn = nn.BatchNorm2d(input_channel)

    def forward(self, x):
        out = self.ca(x)
        out = self.bn(out)
        out = self.sa(out)
        return out