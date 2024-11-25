from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')
from dataset.load_data import  Dataset_test
from model import CSPN
import os
import argparse
torch.backends.cudnn.enabled = False
import torch.nn as nn
from utils import *

#basic setting
parser = argparse.ArgumentParser(description='train_swinir')
parser.add_argument('--mode',default='Test_result')
parser.add_argument('--num_of_patch',default=5,type=int)
parser.add_argument('--gpu',default='1',type=str)
parser.add_argument('--use_amp',default=False)
parser.add_argument('--mgpu_in',default=True)
parser.add_argument('--mgpu_test',default=True)

#parameters of the classification algorithm
parser.add_argument('--alpha',default=torch.tensor(0.3))
#lolv1
parser.add_argument('--center',default=torch.tensor([30.08 ,48.49, 62.86] ))
parser.add_argument('--var',default=torch.tensor([8.07 ,6.37 ,5.65]))
#lolv2
# parser.add_argument('--center',default=torch.tensor([13.56,28.03,50.21]))
# parser.add_argument('--var',default=torch.tensor([7.67,6.88,6.87]))
#lsrw
# parser.add_argument('--center',default=torch.tensor([21.77,39.48,55.52] ))
# parser.add_argument('--var',default=torch.tensor([8.08,7.02,8.21]))

#paths of input and ground truth images
parser.add_argument('--test_root',default='/LOLv1/test/low')
parser.add_argument('--test_gt_root',default='/LOLv1/test/high')

# model parameters
parser.add_argument('--croped_img_size',default=64)
parser.add_argument('--img_size',default=128)
parser.add_argument('--patch_size',default=2)
parser.add_argument('--upscale',default=1)
parser.add_argument('--window_size',default=4)
parser.add_argument('--ape',default=False)
parser.add_argument('--n_feats',default=64)
parser.add_argument('--mlp_ratio',default=1.5)
parser.add_argument('--num_heads',default=4)
parser.add_argument('--drop_path',default=0.5)
parser.add_argument('--attn_drop',default=0.5)

#test setting
parser.add_argument('--saveimg_gap',default=1)
parser.add_argument('--test_freq',default=1)
parser.add_argument('--contine',default=True)
parser.add_argument('--contine_path',default="/data/wuhongjun/project/Test_metric/TCSVT24/weight/lolv2.pth")


def test(t_loader,model,pth,args):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('testing......')
    model = model.eval()
    with torch.no_grad():
        all_psnr=0
        all_ssim=0
        with tqdm(total=len(t_loader)) as tq:
            for idx,(data,gt,name) in enumerate(t_loader):

                data = data.cuda()
                gt = gt.cuda()
                pred = model(data)
                if (idx+1)%args.saveimg_gap==0:
                    torchvision.utils.save_image(pred,pth + '/{}'.format(name[0]))
                # cv2.imwrite(pth + '/{}'.format(name[0]),pred.cpu().numpy()*255.)
                lf=nn.MSELoss()
                PSNR = lambda mse: 10 * torch.log10(1. / mse).item() if (mse > 1e-5) else 50
                psnr=PSNR(lf(gt,pred))
                all_psnr+=psnr

                ssim=SSIM()
                all_ssim+=ssim(gt,pred)

                tq.update()

    return all_psnr/len(t_loader),all_ssim/len(t_loader)

if __name__=='__main__':

    args=parser.parse_args()
    # set_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    dataset_test = Dataset_test(args.test_root, args.test_gt_root)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=16,drop_last=False)

    model=CSPN(args)
    if torch.cuda.is_available():
        model = model.cuda()
        model=nn.DataParallel(model)
    if args.contine:
        params = torch.load(args.contine_path)
        if args.mgpu_in:
            if args.mgpu_test:
                new_dict = {k: v for k, v in params.items()}
            else:
                new_dict = {k[7:]: v for k, v in params.items()}
            result=model.load_state_dict(new_dict,strict=False)
        else:
            if args.mgpu_test:
                new_dict = {'module.'+k: v for k, v in params.items()}
            else:
                new_dict = {k: v for k, v in params.items()}
            result = model.load_state_dict(new_dict,strict=False)
        print(result)
    save_path="./{}".format(args.mode)
    test_psnr,test_ssim = test(test_loader, model, save_path, args)
    print("psnr:{},ssim:{}".format(test_psnr,test_ssim))