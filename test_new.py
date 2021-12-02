# - *- coding: utf- 8 - *-import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
from niidata import *
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import SimpleITK as sitk
import numpy as np
import pandas as pd
import random
import eval_data as ed
import time
from deform import VNet_3d2
from eval_hd import *
import torch.nn.functional as F

args = TestOptions().parse()
if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])

device = torch.device('cuda:{}'.format(0)) if args.gpu_ids else torch.device('cpu')

patients_data = pd.read_csv('xxx.csv')
patients_img = patients_data['img'].tolist()
patients_msk = patients_data['msk'].tolist()

img_list = []
msk_list = []
test_img = []
test_msk = []

for i in range(len(patients_img)):
    img_list.append(patients_img[i])

test_msk = []
test_img = []

for i in range(3):
    img_tmp = []
    msk_tmp = []

    for k in range(5):
        if k == 0:
            img_pre = img_list[i+len(img_list)-3] + '/image_data/t10.nii'
            img_post = img_list[i+len(img_list)-3] + '/image_data/t'+str(k+2)+'.nii'
            img_cur = img_list[i+len(img_list)-3] + '/image_data/t1.nii'
        else:
            img_pre = img_list[i+len(img_list)-3] + '/image_data/t'+str(k)+'.nii'
            img_post = img_list[i+len(img_list)-3] + '/image_data/t'+str(k+2)+'.nii'
            img_cur = img_list[i+len(img_list)-3] + '/image_data/t'+str(k+1)+'.nii'
        #img_cur = img_list[i] + '/image_data/t'+str(k+1)+'.nii'
        msk_cur = img_list[i+len(img_list)-3] + '/label_data/t'+str(k+1)+'_label.nii'
        test_img.append([img_pre,img_cur,img_post])
        #test_img.append(img_cur)
        test_msk.append(msk_cur)

pixel_auc_list = []
mean_auc_list = []
IOU_list = []
dice_list = []
hd_list = []

dir_deform = 'netG-epoch150'

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    #data_loader = CreateDataLoader(opt)
    #dataset = data_loader.load_data()
    accuracy_data = []

    deform_net = VNet_3d2()
    deform_net.to(device)
    deform_net = torch.nn.DataParallel(deform_net, [0])
    tmp = torch.load(dir_deform)
    deform_net.load_state_dict(tmp)

    if opt.second_net == True:
        print('Done')

            
    elif opt.second_net == False:
        model = create_model(opt)
        visualizer = Visualizer(opt)

        crop_filter = sitk.CropImageFilter()
        crop_filter.SetLowerBoundaryCropSize([8,8,0])
        crop_filter.SetUpperBoundaryCropSize([8,8,0])

        for num in range(len(test_img)):
            image_list = test_img[num]
            pre_img = sitk.ReadImage(image_list[0])
            pre_img = sitk.GetArrayFromImage(pre_img)
            pre_img = sitk.GetImageFromArray(pre_img)

            imgdata = sitk.ReadImage(image_list[1])
            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = sitk.GetImageFromArray(imgdata)

            post_img = sitk.ReadImage(image_list[2])
            post_img = sitk.GetArrayFromImage(post_img)
            post_img = sitk.GetImageFromArray(post_img)

            mskdata = sitk.ReadImage(test_msk[num])
            mskdata = sitk.GetArrayFromImage(mskdata)
            mskdata = sitk.GetImageFromArray(mskdata)

            spacing = np.asarray(imgdata.GetSpacing())*np.asarray(imgdata.GetSize(), dtype=float)/[128.0,128.0,96.0]
            factorSize = np.asarray([128,128,96], dtype=int)
            T = sitk.AffineTransform(3)
            T.SetMatrix(imgdata.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(imgdata)
            resampler.SetOutputSpacing(spacing)
            resampler.SetSize(factorSize)
            resampler.SetInterpolator(sitk.sitkBSpline)
            imgdata = resampler.Execute(imgdata)
            pre_img = resampler.Execute(pre_img)
            post_img = resampler.Execute(post_img)

            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float32)
            imgdata = np.clip(imgdata, -500.0, 600.0)
            imgdata = (imgdata - imgdata.mean())/imgdata.std()

            resampler.SetInterpolator(sitk.sitkLabelGaussian)
            mskdata = resampler.Execute(mskdata)
            mskdata = sitk.GetArrayFromImage(mskdata)
            mskdata = mskdata.astype(np.float32)

            pre_img = sitk.GetArrayFromImage(pre_img)
            pre_img = pre_img.astype(np.float32)
            pre_img = np.clip(pre_img, -500.0, 600.0)
            pre_img = (pre_img - pre_img.mean())/pre_img.std()
            post_img = sitk.GetArrayFromImage(post_img)
            post_img = post_img.astype(np.float32)
            post_img = np.clip(post_img, -500.0, 600.0)
            post_img = (post_img - post_img.mean())/post_img.std()

            imgdata = torch.from_numpy(imgdata).unsqueeze(0).unsqueeze(0)
            mskdata = torch.from_numpy(mskdata)#.unsqueeze(0)
            pre_img = torch.from_numpy(pre_img).unsqueeze(0).unsqueeze(0)
            post_img = torch.from_numpy(post_img).unsqueeze(0).unsqueeze(0)
            

            combined_image1 = torch.cat((imgdata,pre_img),1)
            combined_image2 = torch.cat((imgdata,post_img),1)

            epoch_start_time1 = time.time()

            with torch.no_grad():
                motion_field1 = deform_net(combined_image1.to(device))
                motion_field2 = deform_net(combined_image2.to(device))

            epoch_start_time2 = time.time()


            model.set_input(imgdata, motion_field1, motion_field2)


                
            pred, pre_pred, post_pred = model.test_bi()
            pred = F.softmax(pred, dim=1)
            _, pred = torch.max(pred, dim=1)

            print(time.time() - epoch_start_time1)
            print(time.time() - epoch_start_time2)

            pre_pred = F.softmax(pre_pred, dim=1)
            _, pre_pred = torch.max(pre_pred, dim=1)

            post_pred = F.softmax(post_pred, dim=1)
            _, post_pred = torch.max(post_pred, dim=1)

            fused_pred = pre_pred + post_pred
            fused_pred = fused_pred.cpu().data

            fused_pred = torch.where(fused_pred > 1.5, torch.Tensor([1]), torch.Tensor([0]))

            pred = pred.type(torch.FloatTensor)

            dice_tmp = ed.dice_loss(pred, mskdata)
            dice_list.append(dice_tmp.numpy())
            pixel_auc_list.append(ed.pixel_accuracy(pred[0,:,:,:].numpy(), \
                mskdata.data.numpy()))

            mean_auc_list.append(ed.mean_accuracy(pred[0,:,:,:].numpy(), \
                mskdata.data.numpy()))
            IOU_list.append(ed.mean_IU(pred[0,:,:,:].numpy(), \
                mskdata.data.numpy()))
            #hd_eval = Surface(pred[0,:,:,:].numpy().astype(np.int32), \
                #mskdata.data.numpy().astype(np.int32))
            #hd_list.append(hd_eval.hausdorff())

            print(num)

                


    df = pd.DataFrame({'dice':np.array(dice_list).tolist(),'HD':hd_list, 'm_auc':mean_auc_list, \
        'IOU':IOU_list, 'p_auc': pixel_auc_list})
    df.to_csv("our_results_bi.csv", index=False)
    
