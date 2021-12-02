import torch
import torch.nn as nn
import torch.nn.functional as F
from grid_process import Dense3DSpatialTransformer


class VNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2, kernel_sz=3, pad = 1):
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
        self.convs.append(nn.Conv3d( \
                in_channels, out_channels, kernel_size=kernel_sz, padding=pad))
        self.bns.append(nn.BatchNorm3d(out_channels))
        self.afs.append(nn.PReLU(out_channels))
        #self.afs.append(nn.ELU())
        for i in range(self.layers-1):
            self.convs.append(nn.Conv3d( \
                    out_channels, out_channels, kernel_size=kernel_sz, padding=pad))
            self.bns.append(nn.BatchNorm3d(out_channels))
            self.afs.append(nn.PReLU(out_channels))

    def forward(self, x):
        out = x
        for i in range(self.layers):
            out = self.convs[i](out)
            out = self.bns[i](out)
            out = self.afs[i](out)
        return out

class VNetInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=1, k_size=3, pad_size=1):
        super(VNetInBlock, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.convb = VNetConvBlock(in_channels, out_channels, layers=layers, kernel_sz=k_size, pad=pad_size)

    def forward(self, x):
        out = self.bn(x)
        out = self.convb(x)
        #out = torch.add(out, x)
        return out

class VNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(VNetDownBlock, self).__init__()
        self.down = nn.Conv3d( \
                in_channels, out_channels, kernel_size=2, stride=2)
        self.af= nn.PReLU(out_channels)
        self.bn = nn.BatchNorm3d(out_channels)
        self.convb = VNetConvBlock(out_channels, out_channels, layers=layers)

    def forward(self, x):
        down = self.down(x)
        down = self.bn(down)
        down = self.af(down)
        out = self.convb(down)
        out = torch.add(out, down)
        return out

class VNetUpBlock(nn.Module):
    def __init__(self, in_channels, br_channels, out_channels, layers):
        super(VNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
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

class VNetJustUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(VNetJustUpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = VNetConvBlock( \
                out_channels, out_channels, layers=layers)

    def forward(self, x):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        out = self.convb(up)
        #out = torch.add(out, up)
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
        self.conv = nn.Conv3d(in_channels, classes, kernel_size=1)
        self.bn_out = nn.BatchNorm3d(classes)
        self.af_out = nn.PReLU(classes)
        #self.af_out = nn.PReLU(classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn_out(out)
        out = self.af_out(out)
        return out



class VNet_3d(nn.Module):
    def __init__(self, classes_num = 3):
        classes = classes_num
        super(VNet_3d, self).__init__()
        self.in_block = VNetInBlock(2, 32, 1)
        self.down_block1 = VNetDownBlock(32, 32, 2)
        self.down_block2 = VNetDownBlock(32, 64, 3)
        self.down_block3 = VNetDownBlock(64, 128, 3)
        self.down_block4 = VNetDownBlock(128, 256, 3)
        self.down_block5 = VNetDownBlock(256, 512, 3)
        #self.down_block6 = VNetDownBlock(512, 1024, 3)
        #self.up_block1 = VNetUpBlock(1024, 512, 1024, 3)
        self.up_block2 = VNetUpBlock(512, 256, 512, 3)
        self.up_block3 = VNetUpBlock(512, 128, 256, 2)
        self.up_block4 = VNetUpBlock(256, 64, 128, 2)
        self.up_block5 = VNetUpBlock(128, 32, 64, 2)
        self.up_block6 = VNetUpBlock(64, 32, 32, 2)
        self.out_block = VNetOutSingleBlock(32, classes)
        self.warp_layer = Dense3DSpatialTransformer(96,96,96)


    def forward(self, combined_image, move_image, return_features = False):
        br1 = self.in_block(combined_image)
        br2 = self.down_block1(br1)
        br3 = self.down_block2(br2)
        br4 = self.down_block3(br3)
        br5 = self.down_block4(br4)
        out = self.down_block5(br5)
        #out = self.down_block6(br6)
        #out = self.up_block1(out, br6)
        out = self.up_block2(out, br5)
        out = self.up_block3(out, br4)
        out = self.up_block4(out, br3)
        out = self.up_block5(out, br2)
        out = self.up_block6(out, br1)
        
        #out = self.out_block(out, br1)
        if return_features:
            outputs = out
        else:
            field = self.out_block(out)
            outputs = self.warp_layer(move_image, field[:,0,:,:,:], field[:,1,:,:,:], field[:,2,:,:,:])
            #outputs = F.log_softmax(outputs)

        return outputs, field



class VNet_3d1(nn.Module):
    def __init__(self, classes_num = 3):
        classes = classes_num
        super(VNet_3d1, self).__init__()
        self.in_block = VNetInBlock(2, 16, 1)
        self.down_block1 = VNetDownBlock(16, 24, 2)
        self.down_block2 = VNetDownBlock(24, 48, 2)
        self.down_block3 = VNetDownBlock(48, 96, 2)
        self.down_block4 = VNetDownBlock(96, 192, 2)
        #self.down_block5 = VNetDownBlock(192, 384, 2)
        #self.down_block6 = VNetDownBlock(512, 1024, 3)
        #self.up_block1 = VNetUpBlock(1024, 512, 1024, 3)
        #self.up_block2 = VNetUpBlock(384, 192, 384, 2)
        self.up_block3 = VNetUpBlock(192, 96, 128, 2)
        self.up_block4 = VNetUpBlock(128, 48, 80, 2)
        self.up_block5 = VNetUpBlock(80, 24, 48, 2)
        self.up_block6 = VNetUpBlock(48, 16, 24, 2)
        self.out_block = VNetOutSingleBlock(24, classes)
        self.warp_layer = Dense3DSpatialTransformer(96,112,112)
        #self.zeros_data = torch.zeros([1,10,224,224])


    def forward_branch(self, combined_image):
        br1 = self.in_block(combined_image)
        br2 = self.down_block1(br1)
        br3 = self.down_block2(br2)
        br4 = self.down_block3(br3)
        out = self.down_block4(br4)
        #out = self.down_block5(br5)
        #out = self.down_block6(br6)
        #out = self.up_block1(out, br6)
        #out = self.up_block2(out, br5)
        out = self.up_block3(out, br4)
        out = self.up_block4(out, br3)
        out = self.up_block5(out, br2)
        out = self.up_block6(out, br1)
        
        field = self.out_block(out)
        
        #outputs = F.log_softmax(outputs)

        return field

    def forward(self, combined12, combined21, image1, image2):
        field12 = self.forward_branch(combined12)
        field21 = self.forward_branch(combined21)

        fake2 = self.warp_layer(image1, field12[:,0,:,:,:], field12[:,2,:,:,:], field12[:,1,:,:,:])

        fake1 = self.warp_layer(image2, field21[:,0,:,:,:], field21[:,2,:,:,:], field21[:,1,:,:,:])


        return field12, fake2, field21, fake1




class VNet_3d2(nn.Module):
    def __init__(self, classes_num = 3):
        classes = classes_num
        super(VNet_3d2, self).__init__()
        self.in_block = VNetInBlock(2, 16, 1)
        self.down_block1 = VNetDownBlock(16, 24, 2)
        self.down_block2 = VNetDownBlock(24, 48, 2)
        self.down_block3 = VNetDownBlock(48, 96, 2)
        self.down_block4 = VNetDownBlock(96, 192, 2)
        #self.down_block5 = VNetDownBlock(192, 384, 2)
        #self.down_block6 = VNetDownBlock(512, 1024, 3)
        #self.up_block1 = VNetUpBlock(1024, 512, 1024, 3)
        #self.up_block2 = VNetUpBlock(384, 192, 384, 2)
        self.up_block3 = VNetUpBlock(192, 96, 128, 2)
        self.up_block4 = VNetUpBlock(128, 48, 80, 2)
        self.up_block5 = VNetUpBlock(80, 24, 48, 2)
        self.up_block6 = VNetUpBlock(48, 16, 24, 2)
        self.out_block = VNetOutSingleBlock(24, classes)
        self.warp_layer = Dense3DSpatialTransformer(112,112,112)
        #self.zeros_data = torch.zeros([1,10,224,224])


    def forward(self, combined_image):
        br1 = self.in_block(combined_image)
        br2 = self.down_block1(br1)
        br3 = self.down_block2(br2)
        br4 = self.down_block3(br3)
        out = self.down_block4(br4)
        #out = self.down_block5(br5)
        #out = self.down_block6(br6)
        #out = self.up_block1(out, br6)
        #out = self.up_block2(out, br5)
        out = self.up_block3(out, br4)
        out = self.up_block4(out, br3)
        out = self.up_block5(out, br2)
        out = self.up_block6(out, br1)
        
        field = self.out_block(out)
        
        #outputs = F.log_softmax(outputs)

        return field

    '''def forward(self, combined12, combined21, image1, image2):
        field12 = self.forward_branch(combined12)
        field21 = self.forward_branch(combined21)

        fake2 = self.warp_layer(image1, field12[:,0,:,:,:], field12[:,2,:,:,:], field12[:,1,:,:,:])

        fake1 = self.warp_layer(image2, field21[:,0,:,:,:], field21[:,2,:,:,:], field21[:,1,:,:,:])


        return field12, fake2, field21, fake1'''





class VNet_3d_ALL(nn.Module):
    def __init__(self, classes_num = 6):
        classes = classes_num
        super(VNet_3d_ALL, self).__init__()
        self.in_block = VNetInBlock(2, 24, 2)
        self.down_block1 = VNetDownBlock(24, 48, 2)#48*48*48
        self.down_block2 = VNetDownBlock(48, 64, 2)#24*24*24
        self.down_block3 = VNetDownBlock(64, 128, 2)#12*12*12
        self.down_block4 = VNetDownBlock(128, 224, 2)#6*6*6
        #self.down_block6 = VNetDownBlock(512, 1024, 3)
        #self.up_block1 = VNetUpBlock(1024, 512, 1024, 3)
        #self.up_block2 = VNetJustUpBlock(280, 140, 2)
        self.up_block3 = VNetUpBlock(224, 128, 160, 2)#12
        self.up_block4 = VNetUpBlock(160, 64, 100, 2)#24
        #self.out24_1 = VNetInBlock(100, 64, 2)
        self.out24_2 = VNetInBlock(100, 32, 2)
        self.out24_3 = VNetOutSingleBlock(32, classes)
        self.up_block5 = VNetUpBlock(100, 48, 64, 2)#48
        self.out48_1 = VNetInBlock(64, 32, 2)
        self.out48_2 = VNetInBlock(64, 32, 2)
        self.out48_3 = VNetOutSingleBlock(32, classes)
        self.up_block6 = VNetJustUpBlock(64, 32, 2)#96
        self.out96_1 = VNetInBlock(64, 32, 1)
        self.out96_block = VNetOutSingleBlock(32, classes)
        self.warp_layer96 = Dense3DSpatialTransformer(96,112,112)
        self.warp_layer48 = Dense3DSpatialTransformer(48,56,56)
        self.warp_layer24 = Dense3DSpatialTransformer(24,28,28)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')#Interpolate(scale_factor=(2, 2, 2), mode='trilinear')

        

    def forward_branch(self, combined_image):
        br1 = self.in_block(combined_image)
        br2 = self.down_block1(br1)
        br3 = self.down_block2(br2)
        br4 = self.down_block3(br3)
        out = self.down_block4(br4)
        #out = self.down_block6(br6)
        #out = self.up_block1(out, br6)
        out = self.up_block3(out, br4)
        out = self.up_block4(out, br3)
        #out24 = self.out24_1(out)
        out24 = self.out24_2(out)
        deform24 = self.out24_3(out24)
        out = self.up_block5(out, br2)
        out48 = self.out48_1(out)
        out48 = torch.cat((out48, self.upsample(out24)), 1)
        out48 = self.out48_2(out48)
        deform48 = self.out48_3(out48)
        out = self.up_block6(out)
        out96 = torch.cat((out, self.upsample(out48)), 1)
        out96 = self.out96_1(out96)
        
        deform96 = self.out96_block(out96)
        
        #outputs = F.log_softmax(outputs)

        return deform24, deform48, deform96

    def forward(self, combined12, image1_96, image1_48, image1_24, image2_96, image2_48, image2_24, t1, t2):
        field12_24, field12_48, field12_96 = self.forward_branch(combined12)
        inter1_24 = self.warp_layer24(image1_24, t1*field12_24[:,0,:,:,:], t1*field12_24[:,1,:,:,:], t1*field12_24[:,2,:,:,:])
        inter1_48 = self.warp_layer48(image1_48, t1*field12_48[:,0,:,:,:], t1*field12_48[:,1,:,:,:], t1*field12_48[:,2,:,:,:])
        inter1_96 = self.warp_layer96(image1_96, t1*field12_96[:,0,:,:,:], t1*field12_96[:,1,:,:,:], t1*field12_96[:,2,:,:,:])

        inter_field_24_1 = -self.warp_layer24(t2*field12_24[:,3:,:,:,:], t2*field12_24[:,3,:,:,:], t2*field12_24[:,4,:,:,:], t2*field12_24[:,5,:,:,:])
        inter_field_48_1 = -self.warp_layer48(t2*field12_48[:,3:,:,:,:], t2*field12_48[:,3,:,:,:], t2*field12_48[:,4,:,:,:], t2*field12_48[:,5,:,:,:])
        inter_field_96_1 = -self.warp_layer96(t2*field12_96[:,3:,:,:,:], t2*field12_96[:,3,:,:,:], t2*field12_96[:,4,:,:,:], t2*field12_96[:,5,:,:,:])

        fake2_24 = \
            self.warp_layer24(inter1_24, inter_field_24_1[:,0,:,:,:], inter_field_24_1[:,1,:,:,:], inter_field_24_1[:,2,:,:,:])
        fake2_48 = \
            self.warp_layer48(inter1_48, inter_field_48_1[:,0,:,:,:], inter_field_48_1[:,1,:,:,:], inter_field_48_1[:,2,:,:,:])
        fake2_96 = \
            self.warp_layer96(inter1_96, inter_field_96_1[:,0,:,:,:], inter_field_96_1[:,1,:,:,:], inter_field_96_1[:,2,:,:,:])

        inter2_24 = self.warp_layer24(image2_24, t2*field12_24[:,3,:,:,:], t2*field12_24[:,4,:,:,:], t2*field12_24[:,5,:,:,:])
        inter2_48 = self.warp_layer48(image2_48, t2*field12_48[:,3,:,:,:], t2*field12_48[:,4,:,:,:], t2*field12_48[:,5,:,:,:])
        inter2_96 = self.warp_layer96(image2_96, t2*field12_96[:,3,:,:,:], t2*field12_96[:,4,:,:,:], t2*field12_96[:,5,:,:,:])

        inter_field_24_2 = -self.warp_layer24(t1*field12_24[:,:3,:,:,:], t1*field12_24[:,0,:,:,:], t1*field12_24[:,1,:,:,:], t1*field12_24[:,2,:,:,:])
        inter_field_48_2 = -self.warp_layer48(t1*field12_48[:,:3,:,:,:], t1*field12_48[:,0,:,:,:], t1*field12_48[:,1,:,:,:], t1*field12_48[:,2,:,:,:])
        inter_field_96_2 = -self.warp_layer96(t1*field12_96[:,:3,:,:,:], t1*field12_96[:,0,:,:,:], t1*field12_96[:,1,:,:,:], t1*field12_96[:,2,:,:,:])

        fake1_24 = \
            self.warp_layer24(inter2_24, inter_field_24_2[:,0,:,:,:], inter_field_24_2[:,1,:,:,:], inter_field_24_2[:,2,:,:,:])
        fake1_48 = \
            self.warp_layer48(inter2_48, inter_field_48_2[:,0,:,:,:], inter_field_48_2[:,1,:,:,:], inter_field_48_2[:,2,:,:,:])
        fake1_96 = \
            self.warp_layer96(inter2_96, inter_field_96_2[:,0,:,:,:], inter_field_96_2[:,1,:,:,:], inter_field_96_2[:,2,:,:,:])

        

        return fake1_24, fake1_48, fake1_96, fake2_24, fake2_48, fake2_96, field12_96, inter1_96, inter2_96






class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x






