import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import torchvision
import math
import copy

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

#Create Gaussian kernel
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1/v2 )  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


class SSIM_train(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM_train, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return 1-ssim_train(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
def ssim_train(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    # C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1/v2 )  # contrast sensitivity

    return cs

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)
    mse = torch.mean((img1 - img2)**2)
    if mse == 0:
        return float(80)
    return 20 * math.log10(255.0 / math.sqrt(mse))


#用于attention的裁剪
def crop_cpu(img,crop_sz,step):
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        c,h, w = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list = []
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[:,x:x + crop_sz, y:y + crop_sz]
            lr_list.append(crop_img)
    # h = x + crop_sz
    # w = y + crop_sz
    return lr_list, num_h, num_w

def attention_combine(att_maps,images,model):
    kernel=20
    step=10
    pics=images.shape[0]
    results = torch.zeros_like(images)
    if torch.cuda.is_available():
        results=results.cuda()
    images=images.cpu().detach().numpy()
    att_maps = att_maps.cpu().detach().numpy()
    for i in range(pics):
        print('第{}个'.format(i))
        image=images[i,...]
        att_map=att_maps[i,...]
        im_list, num_h, num_w=crop_cpu(image,kernel,step)
        att_list, _, _ = crop_cpu(att_map, kernel, step)
        for j in range(num_h):
            for k in range(num_w):
                tem=torch.from_numpy(im_list[j * num_w + k]).unsqueeze(dim=0)
                if torch.cuda.is_available():
                    tem=tem.cuda()
                avg=np.mean(att_list[j*num_w+k])
                if avg>0.85:
                    result=model(tem,3)

                elif avg>0.8:
                    result=model(tem,2)
                else:
                    result=model(tem,1)
                torchvision.utils.save_image(result, 'result_.png')
                results[i,:,j*step:j*step+kernel,k*step:k*step+kernel]=result.squeeze()
                torchvision.utils.save_image(results[0], 'result.png')
    # for j in range(1, num_w):
    #         results[:,:,:, j * step:j * step + (kernel - step) ]/= 2
    # for k in range(1, num_h):
    #         results[:, :,k * step :k * step  + (kernel - step),:]/= 2
    return results

#颜色损失
def color_loss(x,y):
    b, c, h, w = x.shape
    true_reflect_view = x.view(b, c, h * w).permute(0, 2, 1)
    pred_reflect_view = y.view(b, c, h * w).permute(0, 2, 1)  # 16 x (512x512) x 3
    true_reflect_norm = torch.nn.functional.normalize(true_reflect_view, dim=-1)
    pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
    cose_value = true_reflect_norm * pred_reflect_norm
    cose_value = torch.sum(cose_value, dim=-1)  # 16 x (512x512)  # print(cose_value.min(), cose_value.max())
    color_loss = torch.mean(1 - cose_value)
    return color_loss

def weighted_color_loss(pred,gt):
    b, c, h, w = pred.shape
    true_reflect_view = pred.contiguous().view(b, c, h * w).permute(0, 2, 1)
    pred_reflect_view = gt.contiguous().view(b, c, h * w).permute(0, 2, 1)  # 16 x (512x512) x 3
    true_reflect_norm = torch.nn.functional.normalize(true_reflect_view, dim=-1)
    pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
    cose_value = true_reflect_norm * pred_reflect_norm
    cose_value = torch.sum(cose_value, dim=-1)  # 16 x (512x512)  # print(cose_value.min(), cose_value.max())

    weight=torch.mean(gt,dim=1).unsqueeze(1)
    min_w = torch.min(weight)
    max_w = torch.max(weight)
    weight = (weight - min_w) / (max_w - min_w)
    # a=weight.cpu().numpy()
    color_loss = torch.mean((1 - cose_value)*weight.view(b,h * w))
    return color_loss

def get_glcm_entropy(x,dist,level=64,dir=4):
    x = x[:, 0, ...] * 0.299 + x[:, 1, ...] * 0.587 + x[:, 2, ...] * 0.114
    x = torch.clamp(x * level, max=level - 1e-5).int()
    glcm = torch.zeros([x.shape[0],len(dist)*dir,level * level]).cuda()
    for idx,i in enumerate(dist):
        for j in range(2):
            if j==0:
                hist = (x[:,:-i,:] * level + x[:,i:,:]).flatten(start_dim=1)
            elif j==1:
                hist = (x[ :, :, :-i] * level + x[ :, :, i:]).flatten(start_dim=1)
            for z in range(hist.shape[0]):

                glcm[z,idx*dir+j,:]=torch.histc(hist[z,:].float(),level*level,min=0,max=level*level)

                glcm[z,idx*dir+j,:]=glcm[z,idx*dir+j,:]/torch.sum(glcm[z,idx*dir+j,:])
                glcm[z,idx*dir+j+2,:]=glcm[z,idx*dir+j,:]
    glcm[glcm==0]=1
    result=torch.sum(-glcm*torch.log(glcm),dim=(1,2))
    return result

def classify_by_glcm_entropy(img,center,var,alpha=0.3,alpha_shift=0.2):
    img=torch.pow(img,alpha)
    c_result = torch.zeros([img.shape[0], 3])
    entropy = get_glcm_entropy(img, [2, 4, 8], 64)
    c_total = torch.mean(entropy)
    c_ex = copy.deepcopy(center)
    c_ex += alpha_shift * (c_total - center[1])
    for i in range(3):
        a = torch.pow(entropy - c_ex[i], 2) / (2 * torch.pow(var[i], 2))
        c_result[:, i] = torch.exp(-a) / var[i]
    c_r = torch.argmax(c_result, dim=1)
    return c_r



def check_image_size(window_size, x):
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def crop_piece(cropsize,x):
    # B,C,H,W=x.shape
    B, C, H, W = x.shape
    x=F.unfold(x,kernel_size=cropsize,stride=cropsize)
    x=x.permute(0, 2, 1).view(B, -1, C, cropsize, cropsize)
    fullshape=x.shape
    L_H=int(H/cropsize)
    L_W=int(W/cropsize)
    return x.contiguous().view(-1,C,cropsize,cropsize),fullshape,[B,C,H,W],[L_H,L_W]

def iwt(x_low,x_high):
    r = 2
    in_batch, in_channel, in_height, in_width = x_high.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2-1)), r * in_height, r * in_width
    x1 = x_low
    x2 = x_high[:, 0:out_channel, :, :] / 2
    x3 = x_high[:, out_channel:out_channel * 2, :, :] / 2
    x4 = x_high[:, out_channel * 2:out_channel * 3, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).cuda() #

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

def dwt(x):
    x01 = x[:, 0::2, :, :] / 2
    x02 = x[:, 1::2, :, :] / 2
    x1 = x01[:, :, 0::2, :]
    x2 = x02[:, :, 0::2, :]
    x3 = x01[:, :, 1::2, :]
    x4 = x02[:, :, 1::2, :]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat((x_HL, x_LH, x_HH), -1)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x