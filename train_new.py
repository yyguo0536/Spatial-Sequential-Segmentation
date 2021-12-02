# - *- coding: utf- 8 - *-
import time
from options.train_options import TrainOptions
#from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from niidata import *
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import SimpleITK as sitk
import numpy as np
import pandas as pd
from deform import VNet_3d2
from scipy.misc import imsave
import torch.nn.functional as F

args = TrainOptions().parse()


if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids)

device = torch.device('cuda:{}'.format(0)) if args.gpu_ids else torch.device('cpu')



patients_data = pd.read_csv('xxx.csv')
patients_img = patients_data['img'].tolist()
patients_msk = patients_data['msk'].tolist()

img_list = []
msk_list = []
train_img = []
train_msk = []


for i in range(len(patients_img)):
    if patients_msk[i] == 1:
        img_list.append(patients_img[i])


for i in range(len(img_list)):
    img_tmp = []
    msk_tmp = []

    for k in range(6):
        img_cur = img_list[i] + '/image_data/t'+str(k+1)+'.nii'
        msk_cur = img_list[i] + '/label_data/t'+str(k+1)+'_label.nii'
        #train_img.append([img_pre,img_cur,img_post])
        train_img.append(img_cur)
        train_msk.append(msk_cur)

crop_filter = sitk.CropImageFilter()
crop_filter.SetLowerBoundaryCropSize([8,8,0])
crop_filter.SetUpperBoundaryCropSize([8,8,0])

train_img_list = []
train_msk_list = []
for i in range(len(train_img)):
    img_tmp = sitk.ReadImage(train_img[i])
    img_tmp = sitk.GetArrayFromImage(img_tmp)
    img_tmp = np.clip(img_tmp, -500.0, 600.0)
    img_tmp = (img_tmp - img_tmp.mean())/img_tmp.std()
    img_tmp = sitk.GetImageFromArray(img_tmp)

    msk_tmp = sitk.ReadImage(train_msk[i])
    msk_tmp = sitk.GetArrayFromImage(msk_tmp)
    msk_tmp = sitk.GetImageFromArray(msk_tmp)

    img_tmp = crop_filter.Execute(img_tmp)
    msk_tmp = crop_filter.Execute(msk_tmp)
    #print(imgdata.GetSize())
    spacing = np.asarray(img_tmp.GetSpacing())*np.asarray(img_tmp.GetSize(), dtype=float)/[128.0,128.0,96.0]
    factorSize = np.asarray([128,128,96], dtype=int)
    T = sitk.AffineTransform(3)
    T.SetMatrix(img_tmp.GetDirection())
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img_tmp)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(factorSize)
    resampler.SetInterpolator(sitk.sitkBSpline)
    img_tmp = resampler.Execute(img_tmp)
    #pre_imgdata = resampler.Execute(pre_imgdata)

    resampler.SetInterpolator(sitk.sitkLabelGaussian)
    msk_tmp = resampler.Execute(msk_tmp)

    train_img_list.append(img_tmp)
    train_msk_list.append(msk_tmp)

train_loss = []
#parser = argparse.ArgumentParser(description='GAN + LSTM for Segmentation')

#args = parser.parse_args()

from scipy.misc import imsave
#for i in range((len(training_l)-30),len(training_l)):
    #valid_rows.append([None, training_l[i], label_l[i],None])
dir_deform = 'netG-epoch150'

if __name__ == '__main__':
    opt = TrainOptions().parse() # Define the parameters

    #data_loader = CreateDataLoader(opt) # import the training dataset
    #dataset = data_loader.load_data() # load training dataset
    dataset_size = len(train_img)
    print('#training images = %d' % dataset_size)

    deform_net = VNet_3d2()
    deform_net.to(device)
    deform_net = torch.nn.DataParallel(deform_net, args.gpu_ids)
    tmp = torch.load(dir_deform)
    deform_net.load_state_dict(tmp)

    if opt.second_net == True:
        print('H')

    else:
        model = create_model(opt)
        visualizer = Visualizer(opt)
        total_steps = 0
        total_valid = 0
        train_similar_loss = []



        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0
            epoch_iter_valid = 0

            img_list_tmp = train_img_list#[ii*5:(ii+1)*5]
            msk_list_tmp = train_msk_list#[ii*5:(ii+1)*5]

            rotation_value = (np.random.random(3*9*len(img_list_tmp)/6)-0.5)*0.3*np.pi

            trainning_list = Temporal_cardiac_all_new(\
                    img_list_tmp, msk_list_tmp, job='seg', rotate=rotation_value)

            train_loader = DataLoader(
                        trainning_list, batch_size=args.batchSize, \
                        shuffle=True, num_workers=8, pin_memory=False)

            for batch_idx, (out_dic) in enumerate(train_loader):

                iter_start_time = time.time()
                combined_image12_96 = torch.cat((out_dic['img'], out_dic['ref_img']),1)
                #combined_image21_96 = torch.cat((out_dic['driving'], out_dic['source']),1)
                with torch.no_grad():
                    motion_field = deform_net(combined_image12_96.to(device))

                if total_steps % opt.print_freq == 0:

                    t_data = iter_start_time - iter_data_time

                visualizer.reset()
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                    
                model.set_input(out_dic['img'], motion_field, out_dic['msk'])
                model.optimize_parameters()

                if total_steps % opt.display_freq == 0:

                    save_result = total_steps % opt.update_html_freq == 0
                    #visualizer.display_current_results(model.get_current_visuals(), epoch, epoch_iter, save_result)


                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    #train_gradient_loss.append(losses['G_gradient'])
                    train_loss.append(losses['G_loss'])
                    #train_field_loss.append(losses['G_field'])
                    #train_angle_loss.append(losses['G_angle'])
                    #train_distance_loss.append(losses['G_distance'])
                    #train_similar_loss.append(losses['G24_L1']+losses['G48_L1']+losses['G96_L1'])
                    #train_simiangle_loss.append(losses['G_simiangle'])
                    #train_sum_loss.append(losses['G_sum'])
                    t = (time.time() - iter_start_time) / opt.batchSize
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            iter_data_time = time.time()
            

            plt.plot(train_loss)
            plt.title("simiangle_loss_epoch={}".format(epoch))
            plt.xlabel("Number of iterations")
            plt.ylabel("Average DICE loss per batch")
            plt.savefig("{}/trainloss_epoch={}.png".format(args.checkpoints_dir, 'train_loss'))

            np.save('{}/TrainLoss_epoch={}.npy'.format(args.checkpoints_dir, 'train_loss'),
                                                            np.asarray(train_loss))
            plt.close('all')

            '''plt.plot(train_simiangle_loss)
            plt.title("simiangle_loss_epoch={}".format(epoch))
            plt.xlabel("Number of iterations")
            plt.ylabel("Average DICE loss per batch")
            plt.savefig("{}/trainloss_epoch={}.png".format(args.checkpoints_dir, 'train_simiangle_loss'))

            np.save('{}/TrainLoss_epoch={}.npy'.format(args.checkpoints_dir, 'train_simiangle_loss'),
                                                            np.asarray(train_simiangle_loss))
            plt.close('all')'''


            '''plt.plot(train_sum_loss)
            plt.title("sum_loss_epoch={}".format(epoch))
            plt.xlabel("Number of iterations")
            plt.ylabel("Average DICE loss per batch")
            plt.savefig("{}/trainloss_epoch={}.png".format(args.checkpoints_dir, 'train_sum_loss'))

            np.save('{}/TrainLoss_epoch={}.npy'.format(args.checkpoints_dir, 'train_sum_loss'),
                                                            np.asarray(train_sum_loss))
            plt.close('all')'''


            if epoch % opt.save_epoch_freq == 0:

                print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))

                model.save_net(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

            model.update_learning_rate()






