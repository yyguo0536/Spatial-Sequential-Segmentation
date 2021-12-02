import torch
import torch.nn as nn
import torch.nn.functional as F
from .grid_process import Dense3DSpatialTransformer


class NetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2):
        '''
        ConvBlock = consistent convs
        for each conv, conv(5x5) -> BN -> activation(PReLU)
        params:
        in/out channels: output/input channels
        layers: number of convolution layers
        '''
        super(NetConvBlock, self).__init__()
        self.layers = layers
        self.afs = torch.nn.ModuleList() # activation functions
        self.convs = torch.nn.ModuleList() # convolutions
        self.bns = torch.nn.ModuleList()
        # first conv
        self.convs.append(nn.Conv3d( \
                in_channels, out_channels, kernel_size=3, padding=1))
        self.bns.append(nn.BatchNorm3d(out_channels))
        self.afs.append(nn.PReLU(out_channels))
        #self.afs.append(nn.ELU())
        for i in range(self.layers-1):
            self.convs.append(nn.Conv3d( \
                    out_channels, out_channels, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm3d(out_channels))
            self.afs.append(nn.PReLU(out_channels))

    def forward(self, x):
        out = x
        for i in range(self.layers):
            out = self.convs[i](out)
            out = self.bns[i](out)
            out = self.afs[i](out)
        return out

class NetInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=1):
        super(NetInBlock, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.convb = NetConvBlock(in_channels, out_channels, layers=layers)

    def forward(self, x):
        out = self.bn(x)
        out = self.convb(x)
        #out = torch.add(out, x)
        return out

class NetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(NetDownBlock, self).__init__()
        self.down = nn.Conv3d( \
                in_channels, out_channels, kernel_size=2, stride=2)
        self.af= nn.PReLU(out_channels)
        self.bn = nn.BatchNorm3d(out_channels)
        self.convb = NetConvBlock(out_channels, out_channels, layers=layers)

    def forward(self, x):
        down = self.down(x)
        down = self.bn(down)
        down = self.af(down)
        out = self.convb(down)
        #out = torch.add(out, down)
        return out

class NetUpBlock(nn.Module):
    def __init__(self, in_channels, br_channels, out_channels, layers):
        super(NetUpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = NetConvBlock( \
                out_channels+br_channels, out_channels, layers=layers)

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        out = torch.cat([up, bridge], 1)
        out = self.convb(out)
        #out = torch.add(out, up)
        return out

class NetJustUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(NetJustUpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = NetConvBlock( \
                out_channels, out_channels, layers=layers)

    def forward(self, x):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        #out = torch.cat([up, bridge], 1)
        out = self.convb(up)
        #out = torch.add(out, up)
        return out



class NetUpBlock_DI(nn.Module):
    def __init__(self, in_channels, br_channels1, br_channels2, out_channels, layers):
        super(NetUpBlock_DI, self).__init__()
        self.up = nn.ConvTranspose3d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = NetConvBlock( \
                out_channels+br_channels1+br_channels2, out_channels, layers=layers)

    def forward(self, x, bridge1, bridge2):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        out = torch.cat([up, bridge1, bridge2], 1)
        out = self.convb(out)
        #out = torch.add(out, up)
        return out




class NetOutSingleBlock(nn.Module):
    def __init__(self, in_channels, classes):
        super(NetOutSingleBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, classes, kernel_size=1)
        self.bn_out = nn.BatchNorm3d(classes)
        self.af_out = nn.PReLU(classes)
        #self.af_out = nn.PReLU(classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn_out(out)
        out = self.af_out(out)
        return out





class Net_3d_Scale(nn.Module):
    def __init__(self, classes_num = 1):
        classes = classes_num
        super(Net_3d_Scale, self).__init__()
        self.img_in_block = NetInBlock(1, 24, 2)
        self.img_down_block1 = NetDownBlock(24, 48, 2)
        self.img_down_block2 = NetDownBlock(48, 64, 2)
        self.img_down_block3 = NetDownBlock(64, 128, 2)
        self.img_down_block4 = NetDownBlock(128, 256, 2)

        self.def_in_block = NetInBlock(3, 24, 2)
        self.def_down_block1 = NetDownBlock(24, 48, 2)
        self.def_down_block2 = NetDownBlock(48, 64, 2)
        self.def_down_block3 = NetDownBlock(64, 128, 2)
        self.def_down_block4 = NetDownBlock(128, 256, 2)

        self.up_block3 = NetUpBlock_DI(512, 128, 128, 256, 2)
        self.up_block4 = NetUpBlock_DI(256, 64, 64, 160, 2)
        self.out24_1 = NetInBlock(160, 64, 2)
        self.out24_2 = NetInBlock(64, 32, 2)
        self.out24_3 = NetOutSingleBlock(32, classes)
        self.up_block5 = NetUpBlock_DI(160, 48, 48, 80, 2)
        self.out48_1 = NetInBlock(80, 32, 2)
        #self.out48_2 = NetInBlock(64, 32, 2)
        self.out48_3 = NetOutSingleBlock(32, classes)


        self.up_block6 = NetJustUpBlock(80, 32, 2)
        #self.out96_1 = NetInBlock(64, 32, 1)
        self.out96_block = NetOutSingleBlock(32, classes)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')#Interpolate(scale_factor=(2, 2, 2), mode='trilinear')

        

    def forward(self, image, def96):
        #image = input[:,0,:,:,:].unsqueeze(1)
        #def96 = input[:,1:,:,:,:]
        dis_map = torch.sqrt((def96**2).sum(1).unsqueeze(1)+1e-7)
        img_br1 = self.img_in_block(image)
        dis_map = dis_map / (dis_map.max()+1e-4)
        #dis_map = torch.sigmoid((dis_map-dis_map.mean())/(dis_map.std()+1e-4))
        img_br1 = img_br1 + img_br1 * dis_map
        img_br2 = self.img_down_block1(img_br1)
        img_br3 = self.img_down_block2(img_br2)
        img_br4 = self.img_down_block3(img_br3)
        img_out = self.img_down_block4(img_br4)

        #def96 = F.softsign(def96/2)

        def_br1 = self.def_in_block(def96)
        def_br2 = self.def_down_block1(def_br1)
        def_br3 = self.def_down_block2(def_br2)
        def_br4 = self.def_down_block3(def_br3)
        def_out = self.def_down_block4(def_br4)

        combined_feature = torch.cat([img_out, def_out], 1)
        
        #out = self.down_block6(br6)
        #out = self.up_block1(out, br6)
        out = self.up_block3(combined_feature, img_br4, def_br4)
        out = self.up_block4(out, img_br3, def_br3)
        out24 = self.out24_1(out)
        out24 = self.out24_2(out24)
        msk24 = self.out24_3(out24)
        out = self.up_block5(out, img_br2, def_br2)
        out48 = self.out48_1(out)
        #out48 = out48 + self.upsample(out24)
        msk48 = self.out48_3(out48)
        out96 = self.up_block6(out)
        #out96 = out + self.upsample(out48)
        
        msk96 = self.out96_block(out96)
        msk = self.upsample(self.upsample(msk24)+msk48)+msk96

        
        return msk


class Net_3d_Scale_test(nn.Module):
    def __init__(self, classes_num = 2):
        classes = classes_num
        super(Net_3d_Scale_test, self).__init__()
        self.img_in_block = NetInBlock(1, 24, 2)
        self.img_down_block1 = NetDownBlock(24, 48, 2)
        self.img_down_block2 = NetDownBlock(48, 64, 2)
        self.img_down_block3 = NetDownBlock(64, 128, 2)
        self.img_down_block4 = NetDownBlock(128, 256, 2)

        self.def_in_block = NetInBlock(3, 24, 2)
        self.def_down_block1 = NetDownBlock(24, 48, 2)
        self.def_down_block2 = NetDownBlock(48, 64, 2)
        self.def_down_block3 = NetDownBlock(64, 128, 2)
        self.def_down_block4 = NetDownBlock(128, 256, 2)

        self.up_block3 = NetUpBlock_DI(512, 128, 128, 256, 2)
        self.up_block4 = NetUpBlock_DI(256, 64, 64, 160, 2)
        self.out24_1 = NetInBlock(160, 64, 2)
        self.out24_2 = NetInBlock(64, 32, 2)
        self.out24_3 = NetOutSingleBlock(32, classes)
        self.up_block5 = NetUpBlock_DI(160, 48, 48, 80, 2)
        self.out48_1 = NetInBlock(80, 32, 2)
        #self.out48_2 = NetInBlock(64, 32, 2)
        self.out48_3 = NetOutSingleBlock(32, classes)


        self.up_block6 = NetJustUpBlock(80, 32, 2)
        #self.out96_1 = NetInBlock(64, 32, 1)
        self.out96_block = NetOutSingleBlock(32, classes)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')#Interpolate(scale_factor=(2, 2, 2), mode='trilinear')

        

    

    def forward_single(self, image, def96):
        dis_map = torch.sqrt((def96**2).sum(1).unsqueeze(1)+1e-7)
        img_br1 = self.img_in_block(image)
        dis_map = torch.sigmoid((dis_map-dis_map.mean())/(dis_map.std()+1e-4))
        img_br1 = img_br1 + img_br1 * dis_map
        img_br2 = self.img_down_block1(img_br1)
        img_br3 = self.img_down_block2(img_br2)
        img_br4 = self.img_down_block3(img_br3)
        img_out = self.img_down_block4(img_br4)

        #def96 = torch.softsign(def96/2)

        def_br1 = self.def_in_block(def96)
        def_br2 = self.def_down_block1(def_br1)
        def_br3 = self.def_down_block2(def_br2)
        def_br4 = self.def_down_block3(def_br3)
        def_out = self.def_down_block4(def_br4)

        combined_feature = torch.cat([img_out, def_out], 1)
        
        #out = self.down_block6(br6)
        #out = self.up_block1(out, br6)
        out = self.up_block3(combined_feature, img_br4, def_br4)
        out = self.up_block4(out, img_br3, def_br3)
        out24 = self.out24_1(out)
        out24 = self.out24_2(out24)
        #msk24 = self.out24_3(out24)
        out = self.up_block5(out, img_br2, def_br2)
        out48 = self.out48_1(out)
        #out48 = out48 + self.upsample(out24)
        #out48 = torch.cat((out48, self.upsample(out24)), 1)
        #out48 = self.out48_2(out48)
        #msk48 = self.out48_3(out48)
        out96 = self.up_block6(out)
        #out96 = out96 + self.upsample(out48)
        #out96 = torch.cat((out, self.upsample(out48)), 1)
        #out96 = self.out96_1(out96)
        
        #msk96 = self.out96_block(out96)
        #msk = self.upsample(self.upsample(msk24)+msk48)+msk96

        
        return out96, out48, out24

    def forward(self, image, def96_pre, def96_post):
        
        pre96, pre48, pre24 = self.forward_single(image, def96_pre)
        post96, post48, post24 = self.forward_single(image, def96_post)
        pre_msk24 = self.out24_3((pre24)/2)
        pre_msk48 = self.out48_3((pre48)/2)        
        pre_msk96 = self.out96_block((pre96)/2)

        post_msk24 = self.out24_3((post24)/2)
        post_msk48 = self.out48_3((post48)/2)        
        post_msk96 = self.out96_block((post96)/2)

        msk = self.upsample(self.upsample((pre_msk24+post_msk24)/2)+\
            (pre_msk48+post_msk48)/2)+(pre_msk96+post_msk96)/2
        msk_pre = self.upsample(self.upsample(pre_msk24)+\
            pre_msk48)+pre_msk96
        msk_post = self.upsample(self.upsample(post_msk24)+\
            post_msk48)+post_msk96

        
        return msk, msk_pre, msk_post

    def forward_bk(self, image, def96_pre, def96_post):

        
        pre96, pre48, pre24 = self.forward_single(image, def96_pre)
        post96, post48, post24 = self.forward_single(image, def96_post)
        msk24 = self.out24_3((pre24+post24)/2)
        #out48 = torch.cat((out48, self.upsample(out24)), 1)
        #out48 = self.out48_2(out48)
        msk48 = self.out48_3((pre48+post48)/2)
        #out96 = torch.cat((out, self.upsample(out48)), 1)
        #out96 = self.out96_1(out96)
        
        msk96 = self.out96_block((pre96+post96)/2)
        msk = self.upsample(self.upsample(msk24)+msk48)+msk96

        
        return msk





class Net_3d_Scale_new(nn.Module):
    def __init__(self, classes_num = 1):
        classes = classes_num
        super(Net_3d_Scale_new, self).__init__()
        self.img_in_block = NetInBlock(1, 24, 2)
        self.img_down_block1 = NetDownBlock(24, 48, 2)
        self.img_down_block2 = NetDownBlock(48, 64, 2)
        self.img_down_block3 = NetDownBlock(64, 128, 2)
        self.img_down_block4 = NetDownBlock(128, 256, 2)

        self.def_in_block = NetInBlock(3, 24, 2)
        self.def_down_block1 = NetDownBlock(24, 48, 2)
        self.def_down_block2 = NetDownBlock(48, 64, 2)
        self.def_down_block3 = NetDownBlock(64, 128, 2)
        self.def_down_block4 = NetDownBlock(128, 256, 2)

        self.up_block3 = NetUpBlock_DI(512, 128, 128, 256, 2)
        self.up_block4 = NetUpBlock_DI(256, 64, 64, 160, 2)
        self.out24_1 = NetInBlock(160, 64, 2)
        self.out24_2 = NetInBlock(64, 32, 2)
        self.out24_3 = NetOutSingleBlock(32, classes)
        self.up_block5 = NetUpBlock_DI(160, 48, 48, 80, 2)
        self.out48_1 = NetInBlock(80, 32, 2)
        #self.out48_2 = NetInBlock(64, 32, 2)
        self.out48_3 = NetOutSingleBlock(32, classes)


        self.up_block6 = NetUpBlock_DI(80, 24, 24, 32, 2)
        #self.out96_1 = NetInBlock(64, 32, 1)
        self.out96_block = NetOutSingleBlock(32, classes)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')#Interpolate(scale_factor=(2, 2, 2), mode='trilinear')

        

    def forward(self, image, def96):
        dis_map = torch.sqrt((def96**2).sum(1).unsqueeze(1)+1e-7)
        img_br1 = self.img_in_block(image)
        dis_map = dis_map / (dis_map.max()+1e-4)
        #dis_map = torch.sigmoid((dis_map-dis_map.mean())/(dis_map.std()+1e-4))
        img_br1 = img_br1 + img_br1 * dis_map
        img_br2 = self.img_down_block1(img_br1)
        img_br3 = self.img_down_block2(img_br2)
        img_br4 = self.img_down_block3(img_br3)
        img_out = self.img_down_block4(img_br4)

        #def96 = F.softsign(def96/2)

        def_br1 = self.def_in_block(def96)
        def_br2 = self.def_down_block1(def_br1)
        def_br3 = self.def_down_block2(def_br2)
        def_br4 = self.def_down_block3(def_br3)
        def_out = self.def_down_block4(def_br4)

        combined_feature = torch.cat([img_out, def_out], 1)
        
        #out = self.down_block6(br6)
        #out = self.up_block1(out, br6)
        out = self.up_block3(combined_feature, img_br4, def_br4)
        out = self.up_block4(out, img_br3, def_br3)
        out24 = self.out24_1(out)
        out24 = self.out24_2(out24)
        msk24 = self.out24_3(out24)
        out = self.up_block5(out, img_br2, def_br2)
        out48 = self.out48_1(out)
        #out48 = out48 + self.upsample(out24)
        #out48 = torch.cat((out48, self.upsample(out24)), 1)
        #out48 = self.out48_2(out48)
        msk48 = self.out48_3(out48)
        out96 = self.up_block6(out, img_br1, def_br1)
        #out96 = out96 + self.upsample(out48)
        #out96 = torch.cat((out, self.upsample(out48)), 1)
        #out96 = self.out96_1(out96)
        
        msk96 = self.out96_block(out96)
        msk = self.upsample(self.upsample(msk24)+msk48)+msk96

        
        return msk