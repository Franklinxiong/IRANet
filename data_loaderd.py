from __future__ import print_function, division
import torch
from skimage import io, transform, color
import numpy as np
import math
from torch.utils.data import Dataset
import re
import os


class RescaleT(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # image, label, ref = sample['image'], sample['label'], sample['reflection']
        image, label, depth, depthm = sample['image'], sample['label'], sample['depth'], sample['depthmissing']
        
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        # keep memory in check: downsample before converting the dtype
        # shrink the image first if it is very large
        max_dim = max(h, w)
        if max_dim > 2000:  # downscale when the longer side exceeds 2000 px
            scale_factor = 2000.0 / max_dim
            temp_size = (int(h * scale_factor), int(w * scale_factor))
            
            # do the first resize in uint8 to save memory
            if image.dtype == np.uint8:
                temp_image = transform.resize(image, temp_size, mode='constant', preserve_range=True).astype(np.uint8)
            else:
                temp_image = transform.resize(image, temp_size, mode='constant', preserve_range=True)
            
            # now convert to float32 and normalize
            if temp_image.dtype == np.uint8:
                image = temp_image.astype(np.float32) / 255.0
            else:
                image = temp_image.astype(np.float32)
        else:
            # the image is small enough, convert directly
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
        
        # final resize in float32
        img = transform.resize(image, (self.output_size, self.output_size), mode='constant', preserve_range=False)
        
        # labels are kept in their original range
        if label.dtype != np.float32:
            label = label.astype(np.float32)
        lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0,
                               preserve_range=True)
        
        # depth data
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        dep = transform.resize(depth, (self.output_size, self.output_size), mode='constant', preserve_range=False)
        
        # depth-missing data
        if depthm.dtype != np.float32:
            depthm = depthm.astype(np.float32)
        depm = transform.resize(depthm, (self.output_size, self.output_size), mode='constant', preserve_range=False)

        return {'image': img, 'label': lbl, 'depth': dep, 'depthmissing': depm}


# class Rescale(object):

    # def __init__(self, output_size):
        # assert isinstance(output_size, (int, tuple))
        # self.output_size = output_size

    # def __call__(self, sample):
        # image, label = sample['image'], sample['label']

        # h, w = image.shape[:2]

        # if isinstance(self.output_size, int):
            # if h > w:
                # new_h, new_w = self.output_size * h / w, self.output_size
            # else:
                # new_h, new_w = self.output_size, self.output_size * w / h
        # else:
            # new_h, new_w = self.output_size

        # new_h, new_w = int(new_h), int(new_w)

        # # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image, (new_h, new_w), mode='constant')
        # lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)

        # return {'image': img, 'label': lbl}


# class CenterCrop(object):

    # def __init__(self, output_size):
        # assert isinstance(output_size, (int, tuple))
        # if isinstance(output_size, int):
            # self.output_size = (output_size, output_size)
        # else:
            # assert len(output_size) == 2
            # self.output_size = output_size

    # def __call__(self, sample):
        # image, label = sample['image'], sample['label']

        # h, w = image.shape[:2]
        # new_h, new_w = self.output_size

        # # print("h: %d, w: %d, new_h: %d, new_w: %d"%(h, w, new_h, new_w))
        # assert ((h >= new_h) and (w >= new_w))

        # h_offset = int(math.floor((h - new_h) / 2))
        # w_offset = int(math.floor((w - new_w) / 2))

        # image = image[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
        # label = label[h_offset: h_offset + new_h, w_offset: w_offset + new_w]

        # return {'image': image, 'label': label}


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # image, label, ref = sample['image'], sample['label'], sample['reflection']
        image, label, depth, depthm = sample['image'], sample['label'], sample['depth'], sample['depthmissing']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]
        # ref = ref[top: top + new_h, left: left + new_w]
        depth = depth[top: top + new_h, left: left + new_w]
        depthm = depthm[top: top + new_h, left: left + new_w]

        return {'image': image, 'label': label, 'depth': depth, 'depthmissing': depthm}


class RandomHorizontalFlip(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        # image, label, ref = sample['image'], sample['label'], sample['reflection']
        image, label, depth, depthm = sample['image'], sample['label'], sample['depth'], sample['depthmissing']
        flip_p = np.random.rand()
        if flip_p < self.p:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
            depth = np.flip(depth, axis=1).copy()
            depthm = np.flip(depthm, axis=1).copy()
        return {'image': image, 'label': label, 'depth': depth, 'depthmissing': depthm, 'depthmissing': depthm}


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):

        # image, label, ref = sample['image'], sample['label'], sample['reflection']
        image, label, depth, depthm = sample['image'], sample['label'], sample['depth'], sample['depthmissing']

        tmpLbl = np.zeros(label.shape)

        if np.max(label) < 1e-6:
            label = label
        else:
            label = label / np.max(label)

        # change the color space
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                        np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                        np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                        np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                        np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                        np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                        np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(tmpImg[:, :, 3])
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(tmpImg[:, :, 4])
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(tmpImg[:, :, 5])

        elif self.flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                        np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                        np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                        np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))
        # print('depth.shape', depth.shape)
        
        #if len(depth.shape) > 2:
        #    if np.all(depth[:,:,0] == depth[:,:,1]) and np.all(depth[:,:,0] == depth[:,:,2]):
        #        tmpDep = depth.transpose((2, 0, 1))
        #    else:
        #        print('error transforming depth image')
        #else:
        # tmpDep = np.repeat(depth[:, :, np.newaxis], 3, axis=2).transpose((2, 0, 1))
        tmpDep = np.expand_dims(depth, 0)
        tmpDepm = np.expand_dims(depthm, 0)

        return {'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl),
                'depth': torch.from_numpy(tmpDep), 'depthmissing': torch.from_numpy(tmpDepm)}


class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, dep_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.dep_name_list = dep_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):

        if not os.path.isfile(self.image_name_list[idx]):
            raise FileNotFoundError 
        image = io.imread(self.image_name_list[idx])
        # print('self.image_name_list[idx]', self.image_name_list[idx])

        if 0 == len(self.label_name_list):
            label_3 = np.zeros(image.shape)
        else:
            if not os.path.isfile(self.label_name_list[idx]):
                raise FileNotFoundError 
            label_3 = io.imread(self.label_name_list[idx])


        if not os.path.isfile(self.dep_name_list[idx]):
            raise FileNotFoundError
        # dep = io.imread(self.dep_name_list[idx], as_gray=True)
        depth_uint16 = io.imread(self.dep_name_list[idx], as_gray=True)
        dep = (depth_uint16 / 65535).astype(np.float32)
        depm = (dep==0).astype(int)

        label = np.zeros(label_3.shape[0:2])
        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        img_name = os.path.basename(self.image_name_list[idx])
        search = re.match('^[1-9][0-9][0-9][0-9]?', img_name)
        # is_ref = 0.
        # if search:
            # is_ref = 1.
        
        sample = {'image': image, 'label': label, 'depth': dep, 'depthmissing': depm}
        
        if self.transform:
            sample = self.transform(sample)

        # sample['is_ref'] = is_ref

        return sample
