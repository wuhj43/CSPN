import shutil
import random
from torchvision import datasets,transforms
import torchvision
from torch.autograd import variable
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from tqdm import tqdm
import logging
import random
import sys
import re
import torch
import torch.distributed as dist
import warnings
from utils import *
warnings.filterwarnings('ignore')
from model import *
import csv
from torchvision import models
# from uti.loss_fuc import L1_Charbonnier_loss
from dataset.load_data import  Dataset_train,Dataset_test
import os
import argparse
torch.backends.cudnn.enabled = False
from torch.cuda.amp import autocast,GradScaler
from utils import *
import contextlib # any random number

#basic setting
parser = argparse.ArgumentParser(description='train_swinir')
parser.add_argument('--mode',default='train')
parser.add_argument('--num_of_patch',default=5,type=int)
#training setting
parser.add_argument('--gpu',default='1',type=str)
parser.add_argument('--lr',default=5e-4,help='learning weight',type=float)
parser.add_argument('--batch_size',default=8,type=int)
parser.add_argument('--epochs',default=10000,type=int)
parser.add_argument('--use_amp',default=False)
parser.add_argument('--mgpu_in',default=True)
parser.add_argument('--mgpu_train',default=True)
parser.add_argument('--alpha',default=torch.tensor(0.3))
parser.add_argument('--center',default=torch.tensor([14.55,26.49,40.19]))
parser.add_argument('--var',default=torch.tensor([7.35,6.55,8.43]))

# data path
parser.add_argument('--input_root',default='/LOLv1/train/low')
parser.add_argument('--label_root',default='/LOLv1/train/high')
parser.add_argument('--test_root',default='/LOLv1/test/low')
parser.add_argument('--test_gt_root',default='/LOLv1/test/high')

# swinir
parser.add_argument('--croped_img_size',default=64)
parser.add_argument('--img_size',default=192)
parser.add_argument('--patch_size',default=2)
parser.add_argument('--upscale',default=1)
parser.add_argument('--window_size',default=4)
parser.add_argument('--ape',default=False)
parser.add_argument('--n_feats',default=64)
parser.add_argument('--mlp_ratio',default=1.5)
parser.add_argument('--num_heads',default=4)
parser.add_argument('--drop_path',default=0.5)
parser.add_argument('--attn_drop',default=0.5)

#save setting
parser.add_argument('--psnr_st',default=23,help='psnr standard for saving model')
parser.add_argument('--val_img_size',default=256)
parser.add_argument('--saveimg_gap',default=5)
parser.add_argument('--test_freq',default=5)
parser.add_argument('--contine',default=False)
parser.add_argument('--contine_path',default=" ")
parser.add_argument('--warmup_ep',default=5)
parser.add_argument('--boundary',default=635)


def get_vgg19():
    with torch.no_grad():
        vgg19=models.vgg19(pretrained=True)
        if torch.cuda.is_available():
            vgg19=vgg19.cuda()
        vgg19=vgg19.features
    for idx,param in enumerate(vgg19.parameters()):
        param.requires_grad = False

    vgg19_model_new = list(vgg19.children())[:17]
    vgg19 = nn.Sequential(*vgg19_model_new)
    return vgg19


def test(t_loader,model,ep,pth,max_psnr,args,amp_cm):
    print('testing......')
    model = model.eval()
    with torch.no_grad():
        all_psnr=0
        all_ssim=0
        with tqdm(total=len(t_loader)) as tq:
            for idx,(data,gt,name) in enumerate(t_loader):
                data = data.cuda()
                gt = gt.cuda()
                if args.use_amp:
                    gt=gt.half()
                with amp_cm():
                    pred=model(data)
                    if (idx+1)%args.saveimg_gap==0:
                        torchvision.utils.save_image(pred,pth + '/test_{}_newest_{}.png'.format(args.mode, idx))
                    lf=nn.MSELoss()
                    PSNR = lambda mse: 10 * torch.log10(1. / mse).item() if (mse > 1e-5) else 50
                    psnr=PSNR(lf(gt,pred))
                    all_psnr+=psnr
                    s=ssim(gt,pred)
                    all_ssim+=s
                tq.update()

        return all_psnr/len(t_loader),all_ssim/len(t_loader)

def train(args,train_loader,test_loader,model,opt,scheduler1,scheduler2,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    lf_1=nn.MSELoss()
    L1=nn.L1Loss()
    cl = SSIM_train()
    max_psnr=21.5
    max_ssim=0.8
    vgg19=get_vgg19()
    t=1
    os.makedirs('./record',exist_ok=True)
    scaler = GradScaler()
    amp_cm = autocast if args.use_amp else contextlib.nullcontext
    headers = ['epoch', 'psnr','ssim']
    with open('record/{}.csv'.format(args.mode), 'a', newline='') as f:
        record = csv.writer(f)
        record.writerow(headers)
    for ep in range(args.epochs):
        running_loss=0.0
        if ep<args.boundary:
            scheduler1.step()
        else:
            scheduler2.step()
        with tqdm(total=len(train_loader)) as tq2:
            for train_batch,(data,gt) in enumerate(train_loader):
                if torch.cuda.is_available():
                    if args.use_amp:
                        data=data.half()
                        gt=gt.half()
                        # loc_label=loc_label.half()
                    data=data.cuda()
                    gt=gt.cuda()
                with amp_cm():
                    ori_pred=model(data)
                    feature_pred=vgg19(ori_pred)
                    feature_gt= vgg19(gt)
                    loss=lf_1(ori_pred, gt)+L1(ori_pred, gt)+\
                         20*(L1(feature_pred,feature_gt) / (ori_pred.shape[-1] * ori_pred.shape[-2] ))+\
                         0.5*weighted_color_loss(ori_pred, gt)+0.7*cl(ori_pred,gt)
                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                # for name, parms in model.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                #           ' -->grad_value:', parms.grad)
                opt.zero_grad()
                running_loss += loss.item()

                tq2.set_description(desc="epoch:{}".format(ep+1),refresh=False)
                tq2.update(1)
        now_loss = running_loss / len(train_loader)/t

        test_psnr=0
        test_ssim=torch.tensor(0)
        if ep==args.boundary:
            torch.save(model.state_dict(), save_path + '/record_ep_{}.pth'.format(ep + 1))
        if (ep+1)%args.test_freq==0:
            test_psnr,test_ssim = test(test_loader, model, ep, save_path, max_psnr,args,amp_cm)
            if test_psnr > max_psnr or test_ssim > max_ssim:
                if test_psnr > max_psnr:
                    max_psnr = test_psnr
                if test_ssim > max_ssim:
                    max_ssim = test_ssim
                torch.save(model.state_dict(), save_path  + '/record_ep_{}.pth'.format(ep + 1))
            torch.save(model.state_dict(), save_path + '/newest.pth')
            with open('record/train_record_{}.csv'.format(args.mode), 'a', newline='') as f:
                record = csv.writer(f)
                record.writerow([ep+1, test_psnr,test_ssim.item()])

        output_infos = '\rTrain===> [epoch {}/{}] [loss {:.4f}] [lr: {:.7f}] [test_ssim {:.4f}] [test_psnr {:.4f}] [best_psnr:{:.4f}]'.format(
            ep + 1, args.epochs, now_loss,opt.param_groups[0]['lr'],test_ssim.item(),test_psnr,max_psnr)

        print(output_infos)
        print('-----------------------------------------------')

if __name__=='__main__':

    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dataset_train = Dataset_train(args.input_root, args.label_root,args.img_size)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,num_workers=16,
                                               pin_memory=True,
                                               drop_last=False)

    dataset_test = Dataset_test(args.test_root, args.test_gt_root)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=16,drop_last=False)

    model=CSPN(args)
    if torch.cuda.is_available():
        model = model.cuda()
        model=nn.DataParallel(model)
    if args.contine:
        params = torch.load(args.contine_path)
        if args.mgpu_in:
            if args.mgpu_train:
                new_dict = {k: v for k, v in params.items()}
            else:
                new_dict = {k[7:]: v for k, v in params.items()}
            result=model.load_state_dict(new_dict,strict=False)
        else:
            if args.mgpu_train:
                new_dict = {'module.'+k: v for k, v in params.items()}
            else:
                new_dict = {k: v for k, v in params.items()}
            result = model.load_state_dict(new_dict,strict=False)
        print(result)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(0.9,0.999),eps=1e-7)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 5, 2)
    scheduler2 =torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,320,1)

    save_path="./{}".format(args.mode)
    train(args,train_loader,test_loader, model, opt, scheduler1,scheduler2,save_path)