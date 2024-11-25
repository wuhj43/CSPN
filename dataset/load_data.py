from torch.utils.data import Dataset
import torchvision
import os
import numpy as np
from utils import augment_img
import cv2
import PIL.Image as Image
import random
from skimage import io,filters,exposure
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as tr
from skimage.filters import gaussian

def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    img=img/255.
    # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    """USM sharpening.

    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I


    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    result=soft_mask * sharp + (1 - soft_mask) * img
    # result=cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
    return result*255.

def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)

def uint2tensor3_nodiv(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

class Dataset_train(Dataset):
    def __init__(self, input_root, label_root, fis=256,use_mixup=False):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        self.full_img_size=fis
        self.use_mixup=use_mixup

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img)

class Dataset_train_ape(Dataset):
    def __init__(self, input_root1,input_root2,input_root3, label_root, fis=256,use_mixup=False):
        self.input_root1 = input_root1
        self.input_files1 = os.listdir(input_root1)
        self.input_files1.sort()

        self.input_root2 = input_root2
        self.input_files2 = os.listdir(input_root2)
        self.input_files2.sort()

        self.input_root3 = input_root3
        self.input_files3 = os.listdir(input_root3)
        self.input_files3.sort()

        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        self.full_img_size=fis
        self.use_mixup=use_mixup

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path1 = os.path.join(self.input_root1, self.input_files1[index])
        input_img1 = io.imread(input_img_path1)

        input_img_path2 = os.path.join(self.input_root2, self.input_files2[index])
        input_img2 = io.imread(input_img_path2)

        input_img_path3 = os.path.join(self.input_root3, self.input_files3[index])
        input_img3 = io.imread(input_img_path3)

        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        # if self.use_mixup:
        #     idx_add=torch.randint(0,self.__len__(),size=[1])
        #     lam = np.random.beta(1,1)
        #
        #     input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
        #     label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img1 = input_img1[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        input_img2 = input_img2[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        input_img3 = input_img3[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]

        # mode = random.randint(0, 7)
        # input_img1 = augment_img(input_img1, mode=mode)
        # input_img2 = augment_img(input_img2, mode=mode)
        # input_img3 = augment_img(input_img3, mode=mode)
        # label_img = augment_img(label_img, mode=mode)

        input_img1 = uint2tensor3(input_img1)
        input_img2 = uint2tensor3(input_img2)
        input_img3 = uint2tensor3(input_img3)
        label_img = uint2tensor3(label_img)

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img1 = TF.pad(input_img1, (0, 0, padw, padh), padding_mode='reflect')
            input_img2 = TF.pad(input_img2, (0, 0, padw, padh), padding_mode='reflect')
            input_img3 = TF.pad(input_img3, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img1,input_img2,input_img3, label_img)


class Dataset_test(Dataset):
    def __init__(self, input_root, label_root):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        # self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        # torchvision.utils.save_image(label_img,'gt.png')
        # torchvision.utils.save_image(input_img, 'inpt.png')
        # return (input_img, label_img, self.files[index])
        return (input_img,label_img,self.input_files[index])