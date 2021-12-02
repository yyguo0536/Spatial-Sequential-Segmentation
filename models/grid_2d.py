import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(Module):

    def __init__(self, height, width, device):
        super(SpatialTransformer, self).__init__()
        self.height, self.width= height, width
        self.x_t = np.zeros([self.height, self.width], dtype=np.float32)
        self.y_t = np.zeros([self.height, self.width], dtype=np.float32)
        if device == None:
            device = torch.device('cuda:0')

        x_t = np.matmul(np.ones(shape=np.stack([self.height, 1])),
                        np.transpose(np.expand_dims(np.linspace(0.0,
                                                                self.width -1.0, self.width), 1), [1, 0]))

        y_t = np.matmul(np.expand_dims(np.linspace(0.0, self.height-1.0, self.height), 1),
                        np.ones(shape=np.stack([1, self.width])))

        self.x_t = torch.from_numpy(x_t.astype(np.float32))
        self.y_t = torch.from_numpy(y_t.astype(np.float32))
        self.x_t = self.x_t.to(device)
        self.y_t = self.y_t.to(device)

        self.device = device


    def forward(self, I, dx_t, dy_t):
        #I = torch.unsqueeze(I,1)
        
        bsize = I.shape[0]
        x_mesh = torch.unsqueeze(self.x_t,dim = 0)
        x_mesh = x_mesh.expand(bsize, self.height, self.width).to(self.device)
        y_mesh = torch.unsqueeze(self.y_t,dim = 0)
        y_mesh = y_mesh.expand(bsize, self.height, self.width).to(self.device)

        x_new = dx_t + x_mesh
        y_new = dy_t + y_mesh

        #x_new[dx_t==0]=0
        #y_new[dy_t==0]=0

        I = F.pad(I, (1,1,1,1), 'constant', 0)

        num_batch = I.shape[0]
        channels = I.shape[1]
        height = I.shape[2]
        width = I.shape[3]

        out_height = x_new.shape[1]
        out_width = x_new.shape[2]

        x_new = x_new.unsqueeze(1)
        x_new = x_new.expand(bsize, channels, out_height, out_width)

        y_new = y_new.unsqueeze(1)
        y_new = y_new.expand(bsize, channels, out_height, out_width)

        x = x_new.view(channels, -1)
        y = y_new.view(channels, -1)

        x = x.float() + 1
        y = y.float() + 1

        max_x = width-1.0
        max_y = height-1.0

        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1


        x0 = torch.clamp(x0, min=0, max=max_x)
        x1 = torch.clamp(x1, min=0, max=max_x)
        y0 = torch.clamp(y0, min=0, max=max_y)
        y1 = torch.clamp(y1, min=0, max=max_y)

        dim2 = width
        dim1 = width*height
        

        rep = torch.t(torch.unsqueeze(torch.ones([out_height*out_width]),1)).to(self.device)

        rep = rep.int()

        x_channel = (torch.range(0,channels-1)*dim1).to(self.device)
        x_channel = x_channel.view(-1,1)

        x_channel = torch.mm(x_channel,rep.float())
        base = x_channel.view(channels, -1)


        base_y0 = base.int() + y0*dim2
        base_y1 = base.int() + y1*dim2

        idx_a = base_y0 + x0
        idx_b = base_y0 + x1
        idx_c = base_y1 + x0
        idx_d = base_y1 + x1


        im_flat = I.view(-1,channels)
        im_flat = im_flat.view(-1)
        im_flat = im_flat.float()

        Ia = torch.gather(im_flat, 0, idx_a.view(-1).long())
        Ib = torch.gather(im_flat, 0, idx_b.view(-1).long())
        Ic = torch.gather(im_flat, 0, idx_c.view(-1).long())
        Id = torch.gather(im_flat, 0, idx_d.view(-1).long())

        Ia = Ia.view(channels, -1)
        Ib = Ib.view(channels, -1)
        Ic = Ic.view(channels, -1)
        Id = Id.view(channels, -1)


        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = dx * dy
        wb = dx * (1-dy)
        wc = (1-dx) * dy
        wd = (1-dx) * (1-dy)

        output = wa*Ia + wb*Ib + wc*Ic + wd*Id
        output = output.view(-1, channels, out_height, out_width)
        return torch.flip(output,[2])
        #return output


