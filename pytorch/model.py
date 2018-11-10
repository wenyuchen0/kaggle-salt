
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def same_padding(input, kernel_size = 3, stride= 1):
    input_size = [input.size(-2), input.size(-1)]
    out_size = [(input_size[0] + stride - 1)//stride,(input_size[1] + stride - 1)//stride]
    
    padding = [max(0, (out_size[0] - 1) * stride + (kernel_size - 1)+ 1 - input_size[0]),
     max(0, (out_size[1] - 1) * stride + (kernel_size - 1) + 1 - input_size[1])]
     
    is_odd = [padding[0] % 2 != 0, padding[1] % 2 != 0]
    # padding right column and bottom row if the padding is odd
    if is_odd[0] or is_odd[1]:
        input = F.pad(input ,[0, int(is_odd[1]), 0, int(is_odd[0])])
        
    return F.pad(input, [padding[1]//2, padding[1]//2, padding[0]//2, padding[0]//2])


class ResidualBlock(nn.Module):
    def __init__(self,in_ch, out_ch, r = 4):
        # radio of the senet
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch,in_ch, 3, padding = 1)
        self.bn = nn.BatchNorm2d(in_ch, track_running_stats=False)
        self.abn = ActivatedBatchNorm(in_ch)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_ch, in_ch//r)
        self.fc2 = nn.Linear(in_ch//r, in_ch)
        
    def forward(self, input_block, dropout_ratio = 0, use_senet = True,activation = False):
        x = self.bn(F.relu(input_block))
        x = self.conv(x)
        x = F.dropout2d(x, dropout_ratio, training = self.training)
        x = self.conv(x)
        x = self.bn(F.relu(x))
        # SEnet layer
        if use_senet:
            se = self.globalpool(x)
            se = se.view(se.size()[:2])
            se = F.relu(self.fc1(se))
            se = torch.sigmoid(self.fc2(se))
            se = se.view(*se.size(),1,1)
            x = x*se
        x.add_(input_block)
        if activation:
            x = self.bn(F.relu(x))
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch,r = 4):
        super(Down, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch ,3, padding = 1)
        self.res  = ResidualBlock(out_ch, out_ch,r)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self,input_layer,dropout_ratio = 0.5):
        u_layer =  self.conv(input_layer)
        u_layer = self.res(u_layer, activation = False)
        u_layer = self.res(u_layer, activation = True)
        down_layer = F.dropout2d(self.pool(u_layer), dropout_ratio, training = self.training)
        return u_layer, down_layer

class Mid(nn.Module):
    def __init__(self, in_ch, out_ch,r = 4):
        super(Mid, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch ,3, padding = 1)
        self.res1  = ResidualBlock(out_ch, out_ch,r )
        self.res2  = ResidualBlock(out_ch, out_ch,r )
    
    def forward(self,x, dropout_ratio = 0.5):
        x = self.conv(x)
        x = self.res1(x, activation = False)
        x = self.res2(x, activation = True)
        return x
    
# design deconv layer according to https://distill.pub/2016/deconv-checkerboard/
class Up(nn.Module):
    def __init__(self, in_ch, out_ch,r = 4):
        super(Up, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 4,stride = 2, padding = 1)
        self.conv = nn.Conv2d(out_ch * 2, out_ch, 3, padding = 1)
        self.res = ResidualBlock(out_ch, out_ch, r)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, u_layer, x,dropout_ratio = 0.5):
        x = self.deconv(x)
        x = F.interpolate(x, size = u_layer.size()[-1],mode = 'nearest')
        x = torch.cat((x,u_layer), dim = 1)
        x = F.dropout2d(x, dropout_ratio, training = self.training)
        x = self.conv(x)
        x = self.res(x, activation = False)
        x = self.res(x, activation = True)
        return x


class UNet(nn.Module):
    def __init__(self, start_filters = 16, dropout_ratio = 0.5, r = 4):
        super(UNet, self).__init__()
        self.down1 = Down(1, start_filters * 1,r)
        self.down2 = Down(start_filters * 1, start_filters * 2, r)
        self.down3 = Down(start_filters * 2, start_filters * 4, r)
        self.down4 = Down(start_filters * 4, start_filters * 8, r)
        self.mid = Mid(start_filters * 8, start_filters *16, r)
        self.up4 = Up(start_filters * 16, start_filters * 8, r) 
        self.up3 = Up(start_filters * 8, start_filters * 4, r) 
        self.up2 = Up(start_filters * 4, start_filters * 2, r) 
        self.up1 = Up(start_filters * 2, start_filters * 1, r) 
        self.out = nn.Conv2d(start_filters * 1, 1, 1)
        
    def forward(self,input_layer):
        # 101 -> 50 -> 25 -> 12 -> 6
        u_layer1, down_layer1 = self.down1(input_layer)
        u_layer2, down_layer2 = self.down2(down_layer1)
        u_layer3, down_layer3 = self.down3(down_layer2)
        u_layer4, down_layer4 = self.down4(down_layer3)
        mid_layer = self.mid(down_layer4)
        # 6 -> 12 -> 25 -> 50 -> 101
        up_layer4 = self.up4(u_layer4, mid_layer)
        up_layer3 = self.up3(u_layer3, up_layer4)
        up_layer2 = self.up2(u_layer2, up_layer3)
        up_layer1 = self.up1(u_layer1, up_layer2)
        out_layer = self.out(up_layer1)
        return  out_layer
