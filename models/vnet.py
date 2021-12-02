import torch
import torch.nn as nn
import torch.nn.functional as F
from grid_2d import SpatialTransformer

class VNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2):
        '''
        ConvBlock = consistent convs
        for each conv, conv(5x5) -> BN -> activation(PReLU)
        params:
        in/out channels: output/input channels
        layers: number of convolution layers
        '''
        super(VNetConvBlock, self).__init__()
        self.layers = layers
        self.afs = torch.nn.ModuleList() # activation functions
        self.convs = torch.nn.ModuleList() # convolutions
        self.bns = torch.nn.ModuleList()
        # first conv
        self.convs.append(nn.Conv2d( \
                in_channels, out_channels, kernel_size=3, padding=1))
        self.bns.append(nn.BatchNorm2d(out_channels))
        self.afs.append(nn.PReLU(out_channels))
        #self.afs.append(nn.ELU())
        for i in range(self.layers-1):
            self.convs.append(nn.Conv2d( \
                    out_channels, out_channels, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm2d(out_channels))
            self.afs.append(nn.PReLU(out_channels))

    def forward(self, x):
        out = x
        for i in range(self.layers):
            out = self.convs[i](out)
            out = self.bns[i](out)
            out = self.afs[i](out)
        return out

class VNetInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=1):
        super(VNetInBlock, self).__init__()
        #self.bn = nn.BatchNorm2d(in_channels)
        self.convb = VNetConvBlock(in_channels, out_channels, layers=layers)

    def forward(self, x):
        #out = self.bn(x)
        out = self.convb(x)
        #out = torch.add(out, x)
        return out

class VNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(VNetDownBlock, self).__init__()
        #self.down = nn.Conv2d( \
                #in_channels, out_channels, kernel_size=2, stride=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.af= nn.PReLU(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.convb = VNetConvBlock(in_channels, out_channels, layers=layers)

    def forward(self, x):
        down = self.down(x)
        #down = self.bn(down)
        #down = self.af(down)
        out = self.convb(down)
        #out = torch.add(out, down)
        return out

class VNetUpBlock(nn.Module):
    def __init__(self, in_channels, br_channels, out_channels, layers):
        super(VNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = VNetConvBlock( \
                out_channels+br_channels, out_channels, layers=layers)

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        out = torch.cat([up, bridge], 1)
        out = self.convb(out)
        out = torch.add(out, up)
        return out


class VNetOutBlock(nn.Module):
    def __init__(self, \
            in_channels, br_channels, out_channels, classes, layers=1):
        super(VNetOutBlock, self).__init__()
        self.up = nn.ConvTranspose2d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn_up = nn.BatchNorm2d(out_channels)
        self.af_up= nn.PReLU(out_channels)
        self.convb = VNetConvBlock( \
                out_channels+br_channels, out_channels, layers=layers)
        self.conv = nn.Conv2d(out_channels, classes, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(classes)
        self.af_out= nn.PReLU(classes)

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.bn_up(up)
        up = self.af_up(up)
        out = torch.cat([up, bridge], 1)
        out = self.convb(out)
        out = torch.add(out, up)
        out = self.conv(out)
        out = self.bn_out(out)
        out = self.af_out(out)
        return out


class VNetOutSingleBlock(nn.Module):
    def __init__(self, in_channels, classes):
        super(VNetOutSingleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, classes, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(classes)
        self.af_out = nn.PReLU(classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn_out(out)
        out = self.af_out(out)
        return out



class VNet_back(nn.Module):
    def __init__(self, classes_num = 2):
        classes = classes_num
        super(VNet_back, self).__init__()
        self.in_block = VNetInBlock(1, 32, 1)
        self.down_block1 = VNetDownBlock(32, 32, 2)
        self.down_block2 = VNetDownBlock(32, 64, 2)
        self.down_block3 = VNetDownBlock(64, 128, 2)
        self.down_block4 = VNetDownBlock(128, 256, 2)
        #self.down_block5 = VNetDownBlock(256, 512, 3)
        #self.down_block6 = VNetDownBlock(512, 1024, 3)
        #self.up_block1 = VNetUpBlock(1024, 512, 1024, 3)
        #self.up_block2 = VNetUpBlock(1024, 256, 512, 3)
        self.up_block3 = VNetUpBlock(256, 128, 256, 2)
        self.up_block4 = VNetUpBlock(256, 64, 128, 2)
        self.up_block5 = VNetUpBlock(128, 32, 64, 2)
        self.up_block6 = VNetUpBlock(64, 32, 32, 2)
        self.out_block = VNetOutSingleBlock(32, classes)


    def forward(self, x, return_features = False):
        br1 = self.in_block(x)
        br2 = self.down_block1(br1)
        br3 = self.down_block2(br2)
        br4 = self.down_block3(br3)
        br5 = self.down_block4(br4)
        #br6 = self.down_block5(br5)
        #out = self.down_block6(br6)
        #out = self.up_block1(out, br6)
        #out = self.up_block2(out, br5)
        out = self.up_block3(br5, br4)
        out = self.up_block4(out, br3)
        out = self.up_block5(out, br2)
        out = self.up_block6(out, br1)
        #out = self.out_block(out, br1)
        if return_features:
            outputs = out
        else:
            outputs = self.out_block(out)
            #outputs = F.log_softmax(outputs)

        return outputs


class OF_Net(nn.Module):
    def __init__(self, classes_num = 2, device=None):
        classes = classes_num
        super(OF_Net, self).__init__()
        self.in_block1 = VNetInBlock(1, 64, 2)
        self.in_block2 = VNetInBlock(64, 128, 2)
        self.in_block3 = VNetInBlock(128*3, 128, 2)
        self.down_block1 = VNetDownBlock(128, 256, 2)
        self.down_block2 = VNetDownBlock(256, 512, 2)
        self.down_block3 = VNetDownBlock(512, 1024, 2)
        #self.down_block5 = VNetDownBlock(256, 512, 3)
        #self.down_block6 = VNetDownBlock(512, 1024, 3)
        #self.up_block1 = VNetUpBlock(1024, 512, 1024, 3)
        #self.up_block2 = VNetUpBlock(1024, 256, 512, 3)
        self.up_block4 = VNetUpBlock(1024, 512, 512, 2)
        self.up_block5 = VNetUpBlock(512, 256, 256, 2)
        self.up_block6 = VNetUpBlock(256, 128, 128, 2)
        self.out_block = VNetOutSingleBlock(128, classes)
        self.warp_layer = SpatialTransformer(128,128,device)


    def forward1(self, x, return_features = False):
        br1 = self.in_block1(x)
        outputs = self.in_block2(br1)
        
        #outputs = F.log_softmax(outputs)

        return outputs

    def forward(self, input):
        x = input[:,:3,:,:]
        motion = [input[:,3:5,:,:], input[:,5:,:,:]]
        x_list = []
        for i in range(x.shape[1]):
            x_list.append(self.forward1(x[:,i,:,:].unsqueeze(1)))

        x_list[1] = self.warp_layer(x_list[1], motion[0][:,0,:,:], motion[0][:,1,:,:])
        x_list[2] = self.warp_layer(x_list[2], motion[1][:,0,:,:], motion[1][:,1,:,:])

        x_concat = torch.cat([x_list[0], x_list[1], x_list[2]], dim=1)
        x_concat = self.in_block3(x_concat)

        br2 = self.down_block1(x_concat)
        br3 = self.down_block2(br2)
        br4 = self.down_block3(br3)
        out = self.up_block4(br4, br3)
        out = self.up_block5(out, br2)
        out = self.up_block6(out, x_list[0])
        #out = self.out_block(out, br1)
        outputs = self.out_block(out)
        return outputs




class VNet(nn.Module):
    def __init__(self, classes_num = 2):
        classes = classes_num
        super(VNet, self).__init__()
        self.in_block = VNetInBlock(1, 32, 1)
        self.down_block1 = VNetDownBlock(32, 64, 2)
        self.down_block2 = VNetDownBlock(64, 128, 2)
        self.down_block3 = VNetDownBlock(128, 256, 2)
        self.down_block4 = VNetDownBlock(256, 512, 2)
        #self.down_block5 = VNetDownBlock(256, 512, 3)
        #self.down_block6 = VNetDownBlock(512, 1024, 3)
        #self.up_block1 = VNetUpBlock(1024, 512, 1024, 3)
        #self.up_block2 = VNetUpBlock(1024, 256, 512, 3)
        self.up_block3 = VNetUpBlock(512, 256, 256, 2)
        self.up_block4 = VNetUpBlock(256, 128, 128, 2)
        self.up_block5 = VNetUpBlock(128, 64, 64, 2)
        self.up_block6 = VNetUpBlock(64, 32, 32, 2)
        self.out_block = VNetOutSingleBlock(32, classes)


    def forward(self, x, return_features = False):
        br1 = self.in_block(x)
        br2 = self.down_block1(br1)
        br3 = self.down_block2(br2)
        br4 = self.down_block3(br3)
        br5 = self.down_block4(br4)
        #br6 = self.down_block5(br5)
        #out = self.down_block6(br6)
        #out = self.up_block1(out, br6)
        #out = self.up_block2(out, br5)
        out = self.up_block3(br5, br4)
        out = self.up_block4(out, br3)
        out = self.up_block5(out, br2)
        out = self.up_block6(out, br1)
        #out = self.out_block(out, br1)
        if return_features:
            outputs = out
        else:
            outputs = self.out_block(out)
            #outputs = F.log_softmax(outputs)

        return outputs