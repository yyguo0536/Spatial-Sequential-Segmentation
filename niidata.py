# - *- coding: utf- 8 - *-
import re
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
#import scipy.io as sio
import SimpleITK as sitk
#import torchvision.transforms as tr
import random
from functools import reduce
import cv2
import numpy as np
from itertools import combinations
import random



class Temporal_cardiac_all_new(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, msk_list, job, \
            spacing=None, crop=None, ratio=None, rotate=None, \
            include_slices=None, norm=False):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        self.rotate = rotate
        self.img_list = []
        self.label_list = []
        self.t_dim = [0,0,0]

        self.list_order = []

        for i in range(len(image_list)/6):
            tmp_order = [[i*6, i*6+1],[i*6+1, i*6],[i*6+1, i*6+2],[i*6+2, i*6+1],[i*6+2, i*6+3], \
                [i*6+3, i*6+2],[i*6+3, i*6+4],[i*6+4, i*6+3],[i*6+4, i*6+5]]
            for j in range(len(tmp_order)):
                self.list_order.append(tmp_order[j])
            

        self.rotation_matrix = []
        if not self.rotate is None:
            for k in range(len(self.rotate)/3):
                self.rotation_matrix.append([self.rotate[3*k],self.rotate[3*k+1],self.rotate[3*k+2]])
        # slices
        
        
        for i in range(len(self.list_order)):

            pre_imgdata = image_list[self.list_order[i][1]]

            imgdata = image_list[self.list_order[i][0]]

            mskdata = msk_list[self.list_order[i][0]]


            if not self.rotate is None:
                patient_num = int(i/6)
                center_list = imgdata.GetSize()
                rotation_center = (center_list[0]//2, center_list[1]//2, center_list[2]//2)
                axis_1 = (0,0,1)
                angle_1 = self.rotation_matrix[patient_num][0]
                axis_2 = (0,1,0)
                angle_2 = self.rotation_matrix[patient_num][1]
                axis_3 = (1,0,0)
                angle_3 = self.rotation_matrix[patient_num][2]
                translation = (0,0,0)
                scale_factor = 1.0
                similarity_1 = sitk.Similarity3DTransform(scale_factor, axis_1, angle_1, translation, rotation_center)
                similarity_2 = sitk.Similarity3DTransform(scale_factor, axis_2, angle_2, translation, rotation_center)
                similarity_3 = sitk.Similarity3DTransform(scale_factor, axis_3, angle_3, translation, rotation_center)
                similarity = sitk.Similarity3DTransform(scale_factor, axis_3, angle_3, translation, rotation_center)

                similarity_1 = np.array(similarity_1.GetMatrix()).reshape(3,3).transpose()
                similarity_2 = np.array(similarity_2.GetMatrix()).reshape(3,3).transpose()
                similarity_3 = np.array(similarity_3.GetMatrix()).reshape(3,3).transpose()

                rotation_matrix_all = np.matmul(similarity_1,np.matmul(similarity_2,similarity_3)).reshape(9).tolist()
                mskdata = self.augmentation(similarity, rotation_matrix_all, mskdata, msk=True)
                imgdata = self.augmentation(similarity, rotation_matrix_all, imgdata)
                pre_imgdata = self.augmentation(similarity, rotation_matrix_all, pre_imgdata)
                

            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)
            if norm:
                imgdata = imgdata / 10
            #imgdata = np.clip(imgdata, -500.0, 600.0)

            #imgdata = (imgdata - imgdata.mean())/imgdata.std()

            predata = sitk.GetArrayFromImage(pre_imgdata)
            predata = predata.astype(np.float64)
            #predata = np.clip(predata, -500.0, 600.0)
            #predata = (predata - predata.mean())/predata.std()
            #print(imgdata.std())
            #print(image_list[i])
            self.img_list.append([imgdata, predata])

            labeldata = sitk.GetArrayFromImage(mskdata)
            self.label_list.append(labeldata)

        
        

    def __len__(self):
        return len(self.list_order)

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        #print(self.list_order[index])
        num_list = self.list_order[index]
        #weights_tmp = self.list_weights[index]

        cur_img = self.img_list[index][0].reshape((1,) + self.img_list[index][0].shape)
        pre_img = self.img_list[index][1].reshape((1,) + self.img_list[index][1].shape)

        label_img = self.label_list[index]
        
        
        #size = np.array(image_data.shape)
        # one channel image
        #image_data = image_data.reshape((1,) + image_data.shape)
        out = {}
        cur_img = torch.from_numpy(cur_img.astype(np.float32))
        pre_img = torch.from_numpy(pre_img.astype(np.float32))
        label_img = torch.from_numpy(label_img.astype(np.int64))

        out['img'] = cur_img
        out['ref_img'] = pre_img
        out['msk'] = label_img

        #index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return out

    def augmentation(self, similarity, rotate_matrix, img, msk=False):
        

        affine = sitk.AffineTransform(3)
        affine.SetMatrix(rotate_matrix)
        affine.SetTranslation(similarity.GetTranslation())
        affine.SetCenter(similarity.GetCenter())

        img_tmp = sitk.GetArrayFromImage(img)
        img = sitk.GetImageFromArray(img_tmp)

        if msk:
            imgdata = sitk.Resample(img, img.GetSize(), affine, sitk.sitkLabelGaussian)
        else:
            imgdata = sitk.Resample(img, img.GetSize(), affine, sitk.sitkBSpline)

        return imgdata





class Temporal_cardiac_all(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, msk_list, job, \
            spacing=None, crop=None, ratio=None, rotate=None, \
            include_slices=None, norm=False):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        self.rotate = rotate
        self.img_list = []
        self.label_list = []
        self.t_dim = [0,0,0]

        self.list_order = []

        for i in range(len(image_list)/5):
            tmp_order = [[i*5, i*5+1],[i*5+1, i*5],[i*5+1, i*5+2],[i*5+2, i*5+1],[i*5+2, i*5+3], \
                [i*5+3, i*5+2],[i*5+3, i*5+4],[i*5+4, i*5+3]]
            for j in range(len(tmp_order)):
                self.list_order.append(tmp_order[j])
            

        self.rotation_matrix = []
        if not self.rotate is None:
            for k in range(len(self.rotate)/3):
                self.rotation_matrix.append([self.rotate[3*k],self.rotate[3*k+1],self.rotate[3*k+2]])
        # slices
        crop_filter = sitk.CropImageFilter()
        crop_filter.SetLowerBoundaryCropSize([8,8,0])
        crop_filter.SetUpperBoundaryCropSize([8,8,0])
        
        for i in range(len(self.list_order)):

            pre_imgdata = sitk.ReadImage(image_list[self.list_order[i][1]])
            pre_imgdata = sitk.GetArrayFromImage(pre_imgdata)
            pre_imgdata = sitk.GetImageFromArray(pre_imgdata)

            imgdata = sitk.ReadImage(image_list[self.list_order[i][0]])
            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = sitk.GetImageFromArray(imgdata)

            mskdata = sitk.ReadImage(msk_list[self.list_order[i][0]])
            mskdata = sitk.GetArrayFromImage(mskdata)
            mskdata = sitk.GetImageFromArray(mskdata)

            pre_imgdata = crop_filter.Execute(pre_imgdata)
            imgdata = crop_filter.Execute(imgdata)
            mskdata = crop_filter.Execute(mskdata)
            #print(imgdata.GetSize())
            spacing = np.asarray(imgdata.GetSpacing())*np.asarray(imgdata.GetSize(), dtype=float)/[112.0,112.0,112.0]
            factorSize = np.asarray([112,112,112], dtype=int)
            T = sitk.AffineTransform(3)
            T.SetMatrix(imgdata.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(imgdata)
            resampler.SetOutputSpacing(spacing)
            resampler.SetSize(factorSize)
            resampler.SetInterpolator(sitk.sitkBSpline)
            imgdata = resampler.Execute(imgdata)
            pre_imgdata = resampler.Execute(pre_imgdata)

            resampler.SetInterpolator(sitk.sitkLabelGaussian)
            mskdata = resampler.Execute(mskdata)


            if not self.rotate is None:
                patient_num = int(i/5)
                center_list = imgdata.GetSize()
                rotation_center = (center_list[0]//2, center_list[1]//2, center_list[2]//2)
                axis_1 = (0,0,1)
                angle_1 = self.rotation_matrix[patient_num][0]
                axis_2 = (0,1,0)
                angle_2 = self.rotation_matrix[patient_num][1]
                axis_3 = (1,0,0)
                angle_3 = self.rotation_matrix[patient_num][2]
                translation = (0,0,0)
                scale_factor = 1.0
                similarity_1 = sitk.Similarity3DTransform(scale_factor, axis_1, angle_1, translation, rotation_center)
                similarity_2 = sitk.Similarity3DTransform(scale_factor, axis_2, angle_2, translation, rotation_center)
                similarity_3 = sitk.Similarity3DTransform(scale_factor, axis_3, angle_3, translation, rotation_center)
                similarity = sitk.Similarity3DTransform(scale_factor, axis_3, angle_3, translation, rotation_center)

                similarity_1 = np.array(similarity_1.GetMatrix()).reshape(3,3).transpose()
                similarity_2 = np.array(similarity_2.GetMatrix()).reshape(3,3).transpose()
                similarity_3 = np.array(similarity_3.GetMatrix()).reshape(3,3).transpose()

                rotation_matrix_all = np.matmul(similarity_1,np.matmul(similarity_2,similarity_3)).reshape(9).tolist()
                mskdata = self.augmentation(similarity, rotation_matrix_all, mskdata, msk=True)
                imgdata = self.augmentation(similarity, rotation_matrix_all, imgdata)
                pre_imgdata = self.augmentation(similarity, rotation_matrix_all, pre_imgdata)
                

            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)
            if norm:
                imgdata = imgdata / 10
            imgdata = np.clip(imgdata, -500.0, 600.0)

            imgdata = (imgdata - imgdata.mean())/imgdata.std()

            predata = sitk.GetArrayFromImage(pre_imgdata)
            predata = predata.astype(np.float64)
            predata = np.clip(predata, -500.0, 600.0)
            predata = (predata - predata.mean())/predata.std()
            #print(imgdata.std())
            #print(image_list[i])
            self.img_list.append([imgdata, predata])

            labeldata = sitk.GetArrayFromImage(mskdata)
            self.label_list.append(labeldata)

        
        

    def __len__(self):
        return len(self.list_order)

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        #print(self.list_order[index])
        num_list = self.list_order[index]
        #weights_tmp = self.list_weights[index]

        cur_img = self.img_list[index][0].reshape((1,) + self.img_list[index][0].shape)
        pre_img = self.img_list[index][1].reshape((1,) + self.img_list[index][1].shape)

        label_img = self.label_list[index]
        
        
        #size = np.array(image_data.shape)
        # one channel image
        #image_data = image_data.reshape((1,) + image_data.shape)
        out = {}
        cur_img = torch.from_numpy(cur_img.astype(np.float32))
        pre_img = torch.from_numpy(pre_img.astype(np.float32))
        label_img = torch.from_numpy(label_img.astype(np.int64))

        out['img'] = cur_img
        out['ref_img'] = pre_img
        out['msk'] = label_img

        #index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return out

    def augmentation(self, similarity, rotate_matrix, img, msk=False):
        

        affine = sitk.AffineTransform(3)
        affine.SetMatrix(rotate_matrix)
        affine.SetTranslation(similarity.GetTranslation())
        affine.SetCenter(similarity.GetCenter())

        img_tmp = sitk.GetArrayFromImage(img)
        img = sitk.GetImageFromArray(img_tmp)

        if msk:
            imgdata = sitk.Resample(img, img.GetSize(), affine, sitk.sitkLabelGaussian)
        else:
            imgdata = sitk.Resample(img, img.GetSize(), affine, sitk.sitkBSpline)

        return imgdata



class Temporal_cardiac_all_demons(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, msk_list, def_c_list, def_r_list, job, \
            spacing=None, crop=None, ratio=None, rotate=None, \
            include_slices=None, norm=False):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        self.rotate = rotate
        self.data_list = image_list
        self.img_list = []
        self.label_list = []
        self.def_list_c = []
        self.def_list_r = []
        self.def_list = []
        self.t_dim = [0,0,0]

        self.rotation_matrix = []
        if not self.rotate is None:
            for k in range(len(self.rotate)/3):
                self.rotation_matrix.append([self.rotate[3*k],self.rotate[3*k+1],self.rotate[3*k+2]])
        # slices
        
        
        for i in range(len(image_list)):

            #pre_imgdata = image_list[i]

            imgdata = image_list[i]

            mskdata = msk_list[i]

            pre_deform = def_r_list[i]
            post_deform = def_c_list[i]


            if not self.rotate is None:
                patient_num = int(i/6)
                center_list = imgdata.GetSize()
                rotation_center = (center_list[0]//2, center_list[1]//2, center_list[2]//2)
                axis_1 = (0,0,1)
                angle_1 = self.rotation_matrix[patient_num][0]
                axis_2 = (0,1,0)
                angle_2 = self.rotation_matrix[patient_num][1]
                axis_3 = (1,0,0)
                angle_3 = self.rotation_matrix[patient_num][2]
                translation = (0,0,0)
                scale_factor = 1.0
                similarity_1 = sitk.Similarity3DTransform(scale_factor, axis_1, angle_1, translation, rotation_center)
                similarity_2 = sitk.Similarity3DTransform(scale_factor, axis_2, angle_2, translation, rotation_center)
                similarity_3 = sitk.Similarity3DTransform(scale_factor, axis_3, angle_3, translation, rotation_center)
                similarity = sitk.Similarity3DTransform(scale_factor, axis_3, angle_3, translation, rotation_center)

                similarity_1 = np.array(similarity_1.GetMatrix()).reshape(3,3).transpose()
                similarity_2 = np.array(similarity_2.GetMatrix()).reshape(3,3).transpose()
                similarity_3 = np.array(similarity_3.GetMatrix()).reshape(3,3).transpose()

                rotation_matrix_all = np.matmul(similarity_1,np.matmul(similarity_2,similarity_3)).reshape(9).tolist()
                mskdata = self.augmentation(similarity, rotation_matrix_all, mskdata, msk=True)
                imgdata = self.augmentation(similarity, rotation_matrix_all, imgdata)
                pre_deform = self.augmentation(similarity, rotation_matrix_all, pre_deform)
                post_deform = self.augmentation(similarity, rotation_matrix_all, post_deform)
                

            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)
            if norm:
                imgdata = imgdata / 10
            #imgdata = np.clip(imgdata, -500.0, 600.0)

            #imgdata = (imgdata - imgdata.mean())/imgdata.std()

            pre_deform = sitk.GetArrayFromImage(pre_deform).transpose(3,0,1,2)
            pre_deform = pre_deform.astype(np.float64)

            post_deform = sitk.GetArrayFromImage(post_deform).transpose(3,0,1,2)
            post_deform = post_deform.astype(np.float64)
            #predata = np.clip(predata, -500.0, 600.0)
            #predata = (predata - predata.mean())/predata.std()
            #print(imgdata.std())
            #print(image_list[i])
            self.img_list.append(imgdata)
            self.def_list_c.append(post_deform)
            self.def_list_r.append(pre_deform)

            labeldata = sitk.GetArrayFromImage(mskdata)
            self.label_list.append(labeldata)

        
        

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        #print(self.list_order[index])
        #num_list = self.list_order[index]
        #weights_tmp = self.list_weights[index]

        cur_img = self.img_list[index].reshape((1,) + self.img_list[index].shape)
        pre_img = self.def_list_r[index]
        post_img = self.def_list_c[index]

        label_img = self.label_list[index]
        
        
        #size = np.array(image_data.shape)
        # one channel image
        #image_data = image_data.reshape((1,) + image_data.shape)
        out = {}
        cur_img = torch.from_numpy(cur_img.astype(np.float32))
        pre_img = torch.from_numpy(pre_img.astype(np.float32))
        post_img = torch.from_numpy(post_img.astype(np.float32))
        label_img = torch.from_numpy(label_img.astype(np.int64))

        out['img'] = cur_img
        out['deform_c'] = post_img
        out['deform_r'] = pre_img
        out['msk'] = label_img

        #index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return out

    def augmentation(self, similarity, rotate_matrix, img, msk=False):
        

        affine = sitk.AffineTransform(3)
        affine.SetMatrix(rotate_matrix)
        affine.SetTranslation(similarity.GetTranslation())
        affine.SetCenter(similarity.GetCenter())

        img_tmp = sitk.GetArrayFromImage(img)
        img = sitk.GetImageFromArray(img_tmp)

        if msk:
            imgdata = sitk.Resample(img, img.GetSize(), affine, sitk.sitkLabelGaussian)
        else:
            imgdata = sitk.Resample(img, img.GetSize(), affine, sitk.sitkBSpline)

        return imgdata






class Temporal_lung_all(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, job, \
            spacing=None, crop=None, ratio=None, rotate=True, \
            include_slices=None, norm=False):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        self.rotate = rotate
        self.img_list = []
        self.t_dim = [0,0,0]

        self.list_order = []

        for i in range(int(len(image_list)/5)):
            list_order = [int(i*5), int(i*5+1), int(i*5+2), int(i*5+3), int(i*5+4)]
            list_order = list(combinations(list_order, 2))
            self.list_order = self.list_order + list_order
        #self.list_order = [[0,1],[1,2],[2,3],[3,4]]
        #print(self.list_order)
        self.list_weights = []

        for i in range(len(self.list_order)):
            tmp = 5 - np.abs(self.list_order[i][0]-self.list_order[i][1])
            tmp = (tmp / 4) + 0.8
            self.list_weights.append(tmp)

        if self.rotate:
            rotation_angle = np.random.rand(5)*30.0*np.pi/180.0
            dim_index = np.random.randint(3,size=5)
            #self.t_dim[dim_index] = 1
            #self.t_dim = tuple(self.t_dim)
            #print(rotation_angle)
        
        # slices
        crop_filter = sitk.CropImageFilter()
        crop_filter.SetLowerBoundaryCropSize([32,32,0])
        crop_filter.SetUpperBoundaryCropSize([32,32,0])
        
        for i in range(len(image_list)):
            imgdata = sitk.ReadImage(image_list[i])
            imgdata = crop_filter.Execute(imgdata)
            #print(imgdata.GetSize())
            spacing = np.asarray(imgdata.GetSpacing())*np.asarray(imgdata.GetSize(), dtype=float)/[112.0,112.0,96.0]
            factorSize = np.asarray([112,112,96], dtype=int)
            T = sitk.AffineTransform(3)
            T.SetMatrix(imgdata.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(imgdata)
            resampler.SetOutputSpacing(spacing)
            resampler.SetSize(factorSize)
            resampler.SetInterpolator(sitk.sitkBSpline)
            imgdata = resampler.Execute(imgdata)
            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)
            if norm:
                imgdata = imgdata / 10
            imgdata = np.clip(imgdata, -1000.0, 100.0)

            #print(imgdata.max(), imgdata.min(), imgdata.mean(), imgdata.std())

            imgdata = (imgdata - imgdata.mean())/imgdata.std()
            #print(imgdata.std())
            #print(image_list[i])
            self.img_list.append(imgdata)

        
        

    def __len__(self):
        return len(self.list_order)

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        #print(self.list_order[index])
        num_list = self.list_order[index]
        weights_tmp = self.list_weights[index]

        source_img = self.img_list[num_list[0]].reshape((1,) + self.img_list[num_list[0]].shape)
        driving_img = self.img_list[num_list[1]].reshape((1,) + self.img_list[num_list[1]].shape)
        
        
        #size = np.array(image_data.shape)
        # one channel image
        #image_data = image_data.reshape((1,) + image_data.shape)
        out = {}
        source_img = torch.from_numpy(source_img.astype(np.float32))
        driving_img = torch.from_numpy(driving_img.astype(np.float32))

        out['driving'] = driving_img
        out['source'] = source_img
        out['deformed_weights'] = weights_tmp.astype(np.float32)

        #index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return out




class Analysis_lung_all(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, label_list, job, \
            spacing=None, crop=None, ratio=None, rotate=True, \
            include_slices=None, norm=False):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        self.rotate = rotate
        self.img_list = []
        self.msk_list = []
        self.t_dim = [0,0,0]
        self.inter_t = []

        self.list_order = []

        for i in range(int(len(image_list)/6)):
            for k in range(5):
                list_order = [int(i*6), int(i*6+1+k)]
                
                self.list_order.append(list_order)
                #self.inter_t = self.inter_t + tmp_inter

        
        #self.list_order = [[0,1],[1,2],[2,3],[3,4]]
        #print(self.list_order)
        self.list_weights = []

        #for i in range(len(self.list_order)):
            #self.list_weights.append(np.abs(self.list_order[i][0]-self.list_order[i][1]))

        
        
        # slices
        crop_filter = sitk.CropImageFilter()
        crop_filter.SetLowerBoundaryCropSize([32,32,0])
        crop_filter.SetUpperBoundaryCropSize([32,32,0])
        
        for i in range(len(image_list)):

            tmp_list = []
            imgdata = sitk.ReadImage(image_list[i])
            imgdata = crop_filter.Execute(imgdata)
            #print(imgdata.GetSize())
            spacing = np.asarray(imgdata.GetSpacing())*np.asarray(imgdata.GetSize(), dtype=float)/[112.0,112.0,96.0]
            factorSize = np.asarray([112,112,96], dtype=int)
            T = sitk.AffineTransform(3)
            T.SetMatrix(imgdata.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(imgdata)
            resampler.SetOutputSpacing(spacing)
            resampler.SetSize(factorSize)
            resampler.SetInterpolator(sitk.sitkBSpline)
            imgdata = resampler.Execute(imgdata)
            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)
            if norm:
                imgdata = imgdata / 10
            imgdata = np.clip(imgdata, -1000.0, 100.0)

            #print(imgdata.max(), imgdata.min(), imgdata.mean(), imgdata.std())

            imgdata = (imgdata - imgdata.mean())/imgdata.std()
            self.img_list.append(imgdata)

            imgdata = sitk.ReadImage(label_list[i])
            imgdata = crop_filter.Execute(imgdata)
            #print(imgdata.GetSize())
            spacing = np.asarray(imgdata.GetSpacing())*np.asarray(imgdata.GetSize(), dtype=float)/[112.0,112.0,96.0]
            factorSize = np.asarray([112,112,96], dtype=int)
            T = sitk.AffineTransform(3)
            T.SetMatrix(imgdata.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(imgdata)
            resampler.SetOutputSpacing(spacing)
            resampler.SetSize(factorSize)
            resampler.SetInterpolator(sitk.sitkLabelGaussian)
            imgdata = resampler.Execute(imgdata)
            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)

            self.msk_list.append(imgdata)

        
        

    def __len__(self):
        return (len(self.list_order))

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        #print(self.list_order[index])
        index_list = np.array(self.list_order[index])
        #print(len(self.img_list))
        #print(index_list)

        source_img = self.img_list[index_list[0].astype(int)].reshape((1,) + self.img_list[index_list[0].astype(int)].shape)
        driving_img = self.img_list[index_list[1].astype(int)].reshape((1,) + self.img_list[index_list[1].astype(int)].shape)
        
        source_msk = self.msk_list[index_list[0].astype(int)].reshape((1,) + self.msk_list[index_list[0].astype(int)].shape)
        driving_msk = self.msk_list[index_list[1].astype(int)].reshape((1,) + self.msk_list[index_list[1].astype(int)].shape)
                
        
        #size = np.array(image_data.shape)
        # one channel image
        #image_data = image_data.reshape((1,) + image_data.shape)
        out = {}
        source_img = torch.from_numpy(source_img.astype(np.float32))
        driving_img = torch.from_numpy(driving_img.astype(np.float32))

        source_msk = torch.from_numpy(source_msk.astype(np.float32))
        driving_msk = torch.from_numpy(driving_msk.astype(np.float32))

        out['moving'] = source_img
        out['fixed'] = driving_img

        out['moving_msk'] = source_msk
        out['fixed_msk'] = driving_msk
        #out['deformed_weights'] = weights_tmp.astype(np.float32)

        #index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return out



class Analysis_cardiac_all(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, label_list, job, \
            spacing=None, crop=None, ratio=None, rotate=True, \
            include_slices=None, norm=False):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        self.rotate = rotate
        self.img_list = []
        self.msk_list = []
        self.t_dim = [0,0,0]
        self.inter_t = []

        self.list_order = []

        for i in range(int(len(image_list)/5)):
            for k in range(4):
                list_order = [int(i*5), int(i*5+1+k)]
                
                self.list_order.append(list_order)
                #self.inter_t = self.inter_t + tmp_inter

        
        #self.list_order = [[0,1],[1,2],[2,3],[3,4]]
        #print(self.list_order)
        self.list_weights = []

        #for i in range(len(self.list_order)):
            #self.list_weights.append(np.abs(self.list_order[i][0]-self.list_order[i][1]))

        if self.rotate:
            rotation_angle = np.random.rand(5)*30.0*np.pi/180.0
            dim_index = np.random.randint(3,size=5)
            #self.t_dim[dim_index] = 1
            #self.t_dim = tuple(self.t_dim)
            #print(rotation_angle)
        
        # slices
        crop_filter = sitk.CropImageFilter()
        crop_filter.SetLowerBoundaryCropSize([8,8,0])
        crop_filter.SetUpperBoundaryCropSize([8,8,0])
        
        for i in range(len(image_list)):

            tmp_list = []
            imgdata = sitk.ReadImage(image_list[i])
            imgdata = crop_filter.Execute(imgdata)
            #print(imgdata.GetSize())
            spacing = np.asarray(imgdata.GetSpacing())*np.asarray(imgdata.GetSize(), dtype=float)/[112.0,112.0,112.0]
            factorSize = np.asarray([112,112,112], dtype=int)
            T = sitk.AffineTransform(3)
            T.SetMatrix(imgdata.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(imgdata)
            resampler.SetOutputSpacing(spacing)
            resampler.SetSize(factorSize)
            resampler.SetInterpolator(sitk.sitkBSpline)
            imgdata = resampler.Execute(imgdata)

            if self.rotate:
                img_size = imgdata.GetSize()
                origin = imgdata.GetOrigin()
                t_dim_tmp = self.t_dim
                t_dim_tmp[dim_index[0]] = 1
                t_dim_tmp = tuple(t_dim_tmp)
                transform = sitk.VersorTransform(t_dim_tmp, rotation_angle[0])
                transform.SetCenter((img_size[0]//2, img_size[1]//2, img_size[2]//2))
                imgdata = sitk.Resample(imgdata, imgdata.GetSize(),transform, sitk.sitkLinear, origin, imgdata.GetSpacing(), imgdata.GetDirection())
                imgdata.SetOrigin(origin)

            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)
            if norm:
                imgdata = imgdata / 10
            imgdata = np.clip(imgdata, -500.0, 600.0)

            #print(imgdata.max(), imgdata.min(), imgdata.mean(), imgdata.std())

            imgdata = (imgdata - imgdata.mean())/imgdata.std()
            self.img_list.append(imgdata)

            imgdata = sitk.ReadImage(label_list[i])
            imgdata = crop_filter.Execute(imgdata)
            #print(imgdata.GetSize())
            spacing = np.asarray(imgdata.GetSpacing())*np.asarray(imgdata.GetSize(), dtype=float)/[112.0,112.0,112.0]
            factorSize = np.asarray([112,112,112], dtype=int)
            T = sitk.AffineTransform(3)
            T.SetMatrix(imgdata.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(imgdata)
            resampler.SetOutputSpacing(spacing)
            resampler.SetSize(factorSize)
            resampler.SetInterpolator(sitk.sitkLabelGaussian)
            imgdata = resampler.Execute(imgdata)
            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)

            self.msk_list.append(imgdata)

        
        

    def __len__(self):
        return (len(self.list_order))

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        #print(self.list_order[index])
        index_list = np.array(self.list_order[index])
        #print(len(self.img_list))
        #print(index_list)

        source_img = self.img_list[index_list[0].astype(int)].reshape((1,) + self.img_list[index_list[0].astype(int)].shape)
        driving_img = self.img_list[index_list[1].astype(int)].reshape((1,) + self.img_list[index_list[1].astype(int)].shape)
        
        source_msk = self.msk_list[index_list[0].astype(int)].reshape((1,) + self.msk_list[index_list[0].astype(int)].shape)
        driving_msk = self.msk_list[index_list[1].astype(int)].reshape((1,) + self.msk_list[index_list[1].astype(int)].shape)
                
        
        #size = np.array(image_data.shape)
        # one channel image
        #image_data = image_data.reshape((1,) + image_data.shape)
        out = {}
        source_img = torch.from_numpy(source_img.astype(np.float32))
        driving_img = torch.from_numpy(driving_img.astype(np.float32))

        source_msk = torch.from_numpy(source_msk.astype(np.float32))
        driving_msk = torch.from_numpy(driving_msk.astype(np.float32))

        out['moving'] = source_img
        out['fixed'] = driving_img

        out['moving_msk'] = source_msk
        out['fixed_msk'] = driving_msk
        #out['deformed_weights'] = weights_tmp.astype(np.float32)

        #index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return out







