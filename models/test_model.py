from .base_model import BaseModel
from . import networks
from .cycle_gan_model import CycleGANModel
from .vnet_3d import VNet_3d, UNet_3d
from .unet_3d import UNet3D, PixelDiscriminator

import torch
from .grid_process import Dense3DSpatialTransformer
import torch.nn.functional as F
from .seg_net import Net_3d_Scale, Net_3d_Scale_test
#from thop import profile
#from ptflops import get_model_complexity_info

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser = CycleGANModel.modify_commandline_options(parser, is_train=False)
        parser.set_defaults(dataset_mode='single')
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')

        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [which_epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.warp = Dense3DSpatialTransformer(96,96,96)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G' + opt.model_suffix]

        if opt.which_model_netG == 'motion_seg':
            self.netG = Net_3d_Scale_test(2)
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        elif opt.which_model_netG == 'unet3d':
            #self.netG = UNet3D(1,2)
            self.netG = UNet_3d(2)
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        elif opt.which_model_netG == 'unet2d':
            self.netG = VNet(2)
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        else:
            self.netG = networks.define_G(opt.input_nc*3, opt.output_nc*2, opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)
        if opt.which_model_netG == 'motion_seg':
            self.netG.load_state_dict(torch.load(opt.net3d_dir_G))
        else:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, img, motion_pre, motion_post):
        #AtoB = self.opt.which_direction == 'AtoB'
        #self.real96_A = imgt_96.to(self.device)
        self.real96_A = img.to(self.device)
        self.motion_post = motion_post.to(self.device)
        self.motion_pre = motion_pre.to(self.device)
        #self.index_l = index_l.to(self.device)
        #self.image_paths = input['A_paths']


    def set_input_other(self, img):
        self.real96_A = img.to(self.device)


    def test_other(self):
        with torch.no_grad():
            self.fake_B = self.netG(self.real96_A)
            #flops, params = profile(self.netG,inputs = (self.real96_A))
            #if len(self.real96_A.shape) == 5:
                #flops, params = get_model_complexity_info(self.netG,(1,96,128,128))
            #else:
                #flops, params = get_model_complexity_info(self.netG,(1,128,128))
            print(flops)
            print(params)

        return self.fake_B#self.fake_B#, flops, params

    def test_single(self):
        with torch.no_grad():
            self.fake_B = self.netG(self.real96_A, self.motion_pre)
            #flops, params = get_model_complexity_info(self.netG,(4,96,128,128))
            #$print(flops)
            #print(params)

    
        
        # Calculate final intermediate frame 
        #self.fake_B, _ = torch.max(self.fake_B, dim=1)
        return self.fake_B#, flops, params

    def test_bi(self):
        with torch.no_grad():
            self.fake_B, self.pre, self.post = self.netG(self.real96_A, self.motion_pre, self.motion_post)

        
        
        # Calculate final intermediate frame 
        #self.fake_B, _ = torch.max(self.fake_B, dim=1)
        return self.fake_B, self.pre, self.post
