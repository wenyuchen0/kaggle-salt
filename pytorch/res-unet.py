from same_padding import same_padding
class ResidualBlock(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3)
        self.bn = nn.BatchNorm2d(out_ch)
        
    def forward(self, input_block , combine = "add", dropout_ratio = 0, activation = False):
        x = self.bn(F.relu(input_block))
        x = self.conv(same_padding(x))
        x = F.dropout2d(x, dropout_ratio)
        x = self.conv(same_padding(x))
        x = self.bn(F.relu(x))
        if combine == "add":
            x.add_(input_block)
        if combine == "cat":
            x = torch.cat((x,input_block), dim = 1)
        if activation:
            x = self.bn(F.relu(x))
        return x
 
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch ,3)
        self.res  = ResidualBlock(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self,input_layer,dropout_ratio = 0.5):
        u_layer =  self.conv(same_padding(input_layer))
        u_layer = self.res(u_layer, activation = False)
        u_layer = self.res(u_layer, activation = True)
        down_layer = F.dropout2d(self.pool(u_layer), dropout_ratio)
        return u_layer, down_layer

class Mid(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Mid, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch ,3)
        self.res1  = ResidualBlock(out_ch, out_ch)
        self.res2  = ResidualBlock(out_ch, out_ch)
    
    def forward(self,x, dropout_ratio = 0.5):
        x = self.conv(same_padding(x))
        x = self.res1(x, activation = False)
        x = self.res2(x, activation = True)
        return x
    

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, padding = 0, output_padding = 0):
        super(Up, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 3,stride=2, padding = padding, output_padding = output_padding)
        self.conv = nn.Conv2d(out_ch * 2, out_ch, 3)
        self.res = ResidualBlock(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, u_layer, x,dropout_ratio = 0.5):
        x = self.deconv(x)
        x = torch.cat((x,u_layer), dim = 1)
        x = F.dropout2d(x, dropout_ratio)
        x = self.conv(same_padding(x))
        x = self.res(x, activation = False)
        x = self.res(x, activation = True)
        return x

class Net(nn.Module):
    def __init__(self, start_filters = 16, dropout_ratio = 0.5):
        super(Net, self).__init__()
        self.down1 = Down(1, start_filters * 1)
        self.down2 = Down(start_filters * 1, start_filters * 2)
        self.down3 = Down(start_filters * 2, start_filters * 4)
        self.down4 = Down(start_filters * 4, start_filters * 8)
        self.mid = Mid(start_filters * 8, start_filters *16)
        self.up4 = Up(start_filters * 16, start_filters * 8, 1 , 1)
        self.up3 = Up(start_filters * 8, start_filters * 4, 0, 0)
        self.up2 = Up(start_filters * 4, start_filters * 2, 1, 1)
        self.up1 = Up(start_filters * 2, start_filters * 1, 0, 0)
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
        out_noActivation = self.out(up_layer1)
        out_layer = torch.sigmoid(out_noActivation)
        return  out_noActivation
