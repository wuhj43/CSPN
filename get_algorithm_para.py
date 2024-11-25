# coding = utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import warnings
warnings.filterwarnings("ignore")
img_size=64
from utils import *
from sklearn.mixture import GaussianMixture
import torchvision
import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import tqdm
np.set_printoptions(precision=2)

class MyDataset(Dataset):  # 继承了Dataset子类
    def __init__(self, input_root,transform=None):
        # 分别读取输入/标签图片的路径信息
        self.input_root = input_root
        self.input_files = sorted(os.listdir(input_root))  # 列出指定路径下的所有文件
        # self.train=train
        self.transforms = transform
        # if(self.train):
            # self.input_files.append()

    def __len__(self):
        # 获取数据集大小
        return len(self.input_files)
    # def get_names(self):
    #     return self.input_files
    def __getitem__(self, index):
        # 根据索引(id)读取对应的图片
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = Image.open(input_img_path)
        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        if self.transforms:
            # transforms方法如果有就先处理，然后再返回最后结果
            random.seed(seed)
            input_img = self.transforms(input_img)
        return (input_img,self.input_files[index])

patchsize=64

def ge_glcm_new(name,root):
    num=70 #patch number of every image, change it with different dataset

    dataset = MyDataset(root, transform=torchvision.transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    all_c = torch.zeros(len(data_loader)*num).cuda()
    with tqdm.tqdm(total=len(data_loader)) as tq:
        for idx,(image, _) in enumerate(data_loader):
            x = check_image_size(img_size, image)
            x, fullsize, orisize, L_len = crop_piece(img_size, x)
            x=x.cuda()
            x = x ** (0.3)
            contrary=get_glcm_entropy(x,[2,4,8],img_size)
            all_c[idx*num:(idx+1)*num]=contrary
            # break
            tq.update()

    torch.save(all_c, './{}'.format(name))

def clus(name):
    result = torch.load(name)
    result=result.detach().cpu().numpy()
    g=GaussianMixture(3,covariance_type='full',warm_start=True)
    g.fit(result[:,np.newaxis])
    idx=np.argsort(g.means_,axis=0).squeeze()
    result_center = g.means_.squeeze()[idx]
    result_var=g.covariances_.squeeze()[idx]
    return result_center,np.sqrt(result_var)

if __name__=='__main__':
    name='lolv1.pt'  # GLCM statistical results of the dataset
    root = '/data/wuhongjun/Dataset/LOLv1/train/low' # the path of the dataset
    ge_glcm_new(name,root)
    center,var=clus(name)
    print(center,var)