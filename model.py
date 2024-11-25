from utils import *
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple

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
    return x.contiguous().view(-1,C,cropsize,cropsize),list(fullshape),[B,C,H,W],[L_H,L_W]

def merge_piece(fullsize,orisize,crop_size,x):
    B,L,C,H,W=fullsize
    x=x.view(fullsize).permute(0,2,3,4,1)
    x=x.contiguous().view(B,-1,L)
    x=F.fold(x,output_size=(orisize[-2],orisize[-1]),stride=(crop_size,crop_size),kernel_size=(crop_size,crop_size))
    return x

class CSPN(nn.Module):
    #基于v1 concat adjust取消分组
    def __init__(self,args):
        self.use_amp=args.use_amp
        self.windowsize=args.window_size
        self.dim_num = args.n_feats
        self.alpha = args.alpha.cuda()
        self.alpha_shift = 0.2#args.alpha_shift.cuda()
        self.center=args.center#torch.tensor([14.55,26.49,40.19])
        self.var=args.var#torch.tensor([7.35,6.55,8.43])
        self.train_mode=False
        if torch.cuda.is_available():
            self.center=self.center.cuda()
            self.var = self.var.cuda()
        super(CSPN, self).__init__()
        # self.global_insert=nn.Sequential(Deepse_conv(3,args.n_feats),nn.BatchNorm2d(args.n_feats),nn.ReLU6(),
        #                                   Deepse_conv(args.n_feats,args.n_feats),nn.BatchNorm2d(args.n_feats),nn.ReLU6(),
        #                                   Deepse_conv(args.n_feats,args.n_feats),nn.BatchNorm2d(args.n_feats),nn.ReLU6(),)
        # self.down_sample=nn.AdaptiveAvgPool2d((int(args.croped_img_size/4/args.window_size),int(args.croped_img_size/4/args.window_size)))
        self.branch2=Transformer_RSTB_floor_wi_classify_res(reslution=args.croped_img_size,in_c=args.n_feats,dim=args.n_feats,out_c=args.n_feats,heads=args.num_heads,window_size=args.window_size,patch_size=args.patch_size,drop_path=args.drop_path,attn_drop=args.attn_drop,mlp_ratio=args.mlp_ratio,expansion=6)
        self.crop_size=args.croped_img_size
        self.encoder=Tranformer_encoder_thinv4(self.dim_num,Shuffle_net,1)#Tranformer_encoder(self.dim_num,Shuffle_net)
        self.decoder=Tranformer_decoder_thin_res_v5(self.dim_num,Shuffle_net,Concate_shuffle_res_v5)

    def forward(self,x):
        _,C,W,H=x.shape
        x=check_image_size(self.crop_size,x)
        data_af, fullsize, orisize, L_len = crop_piece(self.crop_size, x)
        cl_result = classify_by_glcm_entropy(data_af,self.center,self.var,self.alpha,self.alpha_shift)#
        # cl_result = torch.randint(low=2, high=3, size=[data_af.shape[0]])
        list_1 = (cl_result == 0).nonzero().squeeze(dim=-1)
        list_2 = (cl_result == 1).nonzero().squeeze(dim=-1)
        list_3 = (cl_result == 2).nonzero().squeeze(dim=-1)
        list_all = [list_1, list_2, list_3]
        x1,x2,x3=self.encoder(data_af)
        b2 = self.branch2(x3, list_all)
        fullsize[2] = self.dim_num
        orisize[1] = self.dim_num
        pred = merge_piece(fullsize, orisize, self.crop_size, b2)
        x1 = merge_piece(fullsize, orisize, self.crop_size, x1)
        x2 = merge_piece(fullsize, orisize, self.crop_size, x2)
        pred = self.decoder(pred,x1,x2)
        pred = pred[:, :, :W, :H]
        return pred

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):  # 判断是否为线性 Linear 层
            nn.init.trunc_normal_(m.weight, std=.02)  # 截断正态分布，限制标准差为 0.02
            if m.bias is not None:  # 如果设置了偏置
                nn.init.constant_(m.bias, 0)  # 初始化偏置为 0
        elif isinstance(m, nn.LayerNorm):  # 判断是否为归一化 LayerNorm 层
            nn.init.constant_(m.bias, 0)  # 初始化偏置为 0
            nn.init.constant_(m.weight, 1.0)


class Tranformer_encoder_thinv4(nn.Module):
    def __init__(self,dim,conv,resdep=1):
        super(Tranformer_encoder_thinv4, self).__init__()
        self.conv1=nn.Conv2d(3, dim, 3,1,padding=1,padding_mode='replicate',bias=True)
        # self.bn=nn.BatchNorm2d(dim)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.res1 = ResBlock(conv, dim, 5, act=nn.GELU(), dep=resdep)
        self.res2 = ResBlock(conv, dim, 5, act=nn.GELU(), dep=resdep)

    def forward(self, x) :
        x1=self.conv1(x)
        x2 = self.res1(x1)
        x3 = self.res2(x2)
        return x1,x2,x3

class Concate_shuffle_res_v5(nn.Module):
    def __init__(self,dim,out_dim=3):
        super(Concate_shuffle_res_v5, self).__init__()
        self.adjust=nn.Conv2d(2*dim,dim,1)
        self.conv1=nn.Sequential(Shuffle_net(dim, dim,1,padding=0,padding_mode='replicate'))
        self.conv2=nn.Sequential(Shuffle_net(dim, dim,3,padding=1,padding_mode='replicate'))#原版只有这里是shuffle
        self.conv3 = nn.Conv1d(1,1,7,padding=3,padding_mode='replicate')
        # self.conv3=nn.Linear(int(out_channel *branch_num),int(out_channel *branch_num))
        if dim==out_dim:
            self.conv4 = nn.Sequential(Shuffle_net(dim, dim,3,padding=1,padding_mode='replicate'), nn.GELU())
        else:
            self.conv4=nn.Sequential(nn.Conv2d(dim,out_dim,3,padding=1,padding_mode='replicate'),nn.GELU())
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.sig=nn.Sigmoid()
        self.relu=nn.GELU()
    def forward(self, x):
        x=self.adjust(x)
        z=self.pool(x).squeeze()
        z=z.unsqueeze(dim=-2)
        if len(z.shape)==2:
            z=z.unsqueeze(dim=0)
        z=self.conv3(z).transpose(-1, -2).unsqueeze(dim=-1)
        # if len(z.shape) == 1:
        #     z.unsqueeze(0)
        # z=self.conv3(z).unsqueeze(-1).unsqueeze(-1)
        return self.conv4(self.relu((self.conv1(x)+self.conv2(x))*self.sig(z)))#

class Tranformer_decoder_thin_res_v5(nn.Module):
    def __init__(self, dim,conv=nn.Conv2d,concat=Concate_shuffle_res_v5):
        super(Tranformer_decoder_thin_res_v5, self).__init__()
        # self.conv1 = nn.Conv2d(3, dim, 3, 1, padding=1, padding_mode='replicate')
        self.res1 = ResBlock(conv, dim, 5, act=nn.GELU(),dep=1,bn=False)
        self.concat1=concat(dim,dim)
        self.res2 = ResBlock(conv, dim, 5, act=nn.GELU(),dep=1,adjust_c=0,bn=False)
        self.concat2 = concat(dim, 3)
    def forward(self, x,x1,x2=None):
        x = self.res1(x)
        x=self.concat1(torch.cat([x,x2],dim=1))
        x=self.res2(x)
        x=self.concat2(torch.cat([x,x1],dim=1))
        return x



class PatchMerging_dwt(nn.Module):
    def __init__(self,input_resolution,  dim, norm_layer=nn.LayerNorm):
        super().__init__()
        if  isinstance(input_resolution,tuple):
            self.input_resolution = input_resolution
        else:
            self.input_resolution = to_2tuple(input_resolution)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 1 * dim, bias=False)
        if norm_layer==None:
            self.norm_high=None
            self.norm_low=None
            # self.norm_low = norm_layer(dim)
            # self.norm_high = norm_layer(3 * dim)
        else:
            self.norm_low = norm_layer( dim)
            self.norm_high = norm_layer(3 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x_low,x_high=dwt(x)
        x_low=x_low.view(B, -1, C)
        x_high = x_high.view(B, -1, 3*C)
        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        #
        # x = self.norm(x)
        # x = self.reduction(x)
        if self.norm_high!=None and self.norm_low!=None:
            x_low=self.norm_low(x_low)
            x_high=self.norm_high(x_high)
        return x_low,x_high

class PatchEmbed(nn.Module):
    def __init__(self, img_size=60, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size,img_size) # -> (img_size, img_size)
        patch_size = (patch_size,patch_size) # -> (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,padding_mode='replicate')
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 假设采取默认参数
        x = self.proj(x) # 出来的是(N, 96, 224/4, 224/4)
        x = torch.flatten(x, 2) # 把HW维展开，(N, 96, 56*56)
        x = torch.transpose(x, 1, 2)  # 把通道维放到最后 (N, 56*56, 96)
        if self.norm is not None:
            x = self.norm(x)
        return x

class WindowAttention_prior(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        # self.layer_norm = nn.LayerNorm(head_dim, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

        torch.nn.init.kaiming_normal(self.relative_position_bias_table)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # q = q * self.scale# B, num_heads, num_window(N), C
        # q=self.layer_norm(q)
        # k = self.layer_norm(k)
        # v = self.layer_norm(v)
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # relative_position_bias=torch.cat([relative_position_bias,torch.ones([relative_position_bias.shape[0],1,relative_position_bias.shape[-1]]).cuda()],dim=1)
        # relative_position_bias = torch.cat([relative_position_bias, torch.ones(
        #     [relative_position_bias.shape[0],relative_position_bias.shape[-2], 1 ]).cuda()], dim=2)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock_RSTB_preln(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_prior(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.relu=nn.ReLU()
        self.register_buffer("attn_mask", attn_mask)
        self.ln_diff=nn.LayerNorm([self.window_size,self.window_size])
        # self.kernel=1/8*torch.FloatTensor([[1,1,1],[1,0,1],[1,1,1]]).unsqueeze(0).unsqueeze(0).cuda()
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # tmp=attn_mask.cpu().numpy()
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        # tem=F.pad(x_windows.view(-1,C, self.window_size , self.window_size),pad=(1,1,1,1),mode='replicate')
        # diff_attn=4*tem[:,:,1:self.window_size+1,1:self.window_size+1]-tem[:,:,:self.window_size,:self.window_size]-tem[:,:,2:,:self.window_size]-tem[:,:,:self.window_size,2:]-tem[:,:,2:,2:]
        # diff_attn = self.ln_diff(diff_attn)
        # diff_attn = diff_attn.view(B, H * W, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # a=torch.zeros_like(x_windows)[:,0,:].unsqueeze(dim=1)
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.relu(self.mlp(self.norm2(x))))#-self.relu(self.mlp_noise(x)*diff_attn)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class BasicLayer_preln(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_RSTB_preln(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchEmbed_swinir(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        if not isinstance(img_size,tuple):
            img_size = to_2tuple(img_size)
        if not isinstance(patch_size, tuple):
            patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops

class PatchUnEmbed_swinir(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

class RSTB_floor_preln(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='3conv',out_dim=0,expansion=4):
        super(RSTB_floor_preln, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer_preln(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            if out_dim==0:
                # self.conv = nn.Conv2d(dim, dim, 3, 1, 1,padding_mode='replicate')
                self.conv = nn.Sequential(nn.BatchNorm2d(dim),nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='replicate'),nn.LeakyReLU(negative_slope=0.2,))
            else:
                self.conv = nn.Conv2d(dim, out_dim, 3, 1, 1, padding_mode='replicate')
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1,padding_mode='replicate'), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0,padding_mode='replicate'),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1,padding_mode='replicate'))
        if out_dim==0:
            self.patch_embed = PatchEmbed_swinir(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)
        else:
            self.patch_embed = PatchEmbed_swinir(
                img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=out_dim,
                norm_layer=None)
        self.patch_unembed = PatchUnEmbed_swinir(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):

        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
         bn=True, act=nn.ReLU(True), res_scale=1,dep=2,adjust_c=0,gr=1):

        super(ResBlock, self).__init__()
        m = []
        self.adjust=None
        for i in range(dep):
            if gr==1:
                m.append(conv(n_feats, n_feats, kernel_size,padding=(kernel_size//2),padding_mode='replicate'))
            else:
                m.append(conv(n_feats, n_feats, kernel_size, padding=(kernel_size // 2), padding_mode='replicate',
                              groups=gr))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        if adjust_c!=0:
            self.adjust=nn.Conv2d(n_feats,adjust_c, kernel_size,padding=(kernel_size//2),padding_mode='replicate')
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        if self.adjust:
            res=self.adjust(res)
        return res

class Shuffle_net(nn.Module):
    def __init__(self,in_c,out_c,kernel,stride=1,padding=1,padding_mode='replicate',group=2):
        in_c=int(in_c/2)
        out_c=out_c-in_c
        super(Shuffle_net, self).__init__()
        self.groups=group
        self.branch = nn.Sequential(
            nn.Conv2d(
                in_c,
                in_c,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(in_c),
            nn.GELU(),
            nn.Conv2d(in_c, in_c, kernel_size=kernel, stride=stride, padding=padding,padding_mode=padding_mode,groups=in_c),
            nn.BatchNorm2d(in_c),
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )
    def forward(self,x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((x1, self.branch(x2)), dim=1)
        out = self.channel_shuffle(out)
        return out

    def channel_shuffle(self,x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x

class RSTB_floor_preln_res_qkv(nn.Module):
    #基于RSTB_floor_preln改过来的卷积qkv
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='3conv',out_dim=0,expansion=4,resdep=8,groups=1):
        super(RSTB_floor_preln_res_qkv, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group =ResBlock(Shuffle_net,dim,5, act=nn.GELU(), dep=resdep)#MBConv(dim,dim,expansion=expansion)

        if resi_connection == '1conv':
            if out_dim==0:
                self.conv = nn.Sequential(nn.BatchNorm2d(dim),nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='replicate'),nn.LeakyReLU(negative_slope=0.2,))
            else:
                self.conv = nn.Conv2d(dim, out_dim, 3, 1, 1, padding_mode='replicate')
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1,padding_mode='replicate',groups=groups), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0,padding_mode='replicate',groups=groups),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1,padding_mode='replicate',groups=groups))

        if out_dim==0:
            self.patch_embed = PatchEmbed_swinir(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)
        else:
            self.patch_embed = PatchEmbed_swinir(
                img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=out_dim,
                norm_layer=None)
        self.patch_unembed = PatchUnEmbed_swinir(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):

        return self.patch_embed(self.residual_group(self.patch_unembed(x, x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

class UnPatchMerging_dwt(nn.Module):
    def __init__(self,input_resolution,dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.unpooling=nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                     # nn.Conv2d(dim,int(dim/2),1,1,0),
                                     )
        self.dim = dim
        # self.production =torch.nn.Identity() #nn.Linear(int(dim/4), int(dim/2), bias=False)
        if norm_layer==None:
            self.norm=None
        else:
            self.norm = norm_layer(self.dim)

    def forward(self, x_low,x_high):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x_low.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x_low = x_low.view(B, H, W, C).permute(0,3,1,2)
        x_high = x_high.view(B, H, W, 3*C).permute(0, 3, 1, 2)
        x=iwt(x_low,x_high)
        x = x.view(B, -1, C)
        # x=self.unpooling(x)
        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # x = x.view(B, -1, C)  # B H/2*W/2 4*C
        if self.norm!=None:
            x = self.norm(x)
        # x = self.production(x)

        return x

class DePatchEmbed(nn.Module):
    def __init__(self, img_size=60, patch_size=4, in_chans=3,out_chan=3,norm_layer=None,p_len=3,use_bn=False):
        super().__init__()
        if not isinstance(img_size,tuple):
            img_size = (img_size,img_size) # -> (img_size, img_size)
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size,patch_size) # -> (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        if use_bn:
            self.unproj =nn.Sequential(nn.BatchNorm2d(in_chans),nn.Conv2d(in_chans,out_chan*patch_size[0]*patch_size[1],kernel_size=(1,1),stride=(1,1),padding_mode='replicate'))
        else:
            self.unproj = nn.Conv2d(in_chans,out_chan*patch_size[0]*patch_size[1],kernel_size=(1,1),stride=(1,1),padding_mode='replicate')
        self.norm_layer=norm_layer
        self.p_len=p_len
        # self.norm=nn.LayerNorm(in_chans)
        self.unpooling=nn.PixelShuffle(patch_size[0])
        self.pad=nn.ReflectionPad2d(p_len)
    def forward(self, x):
        # 假设采取默认参数
        # if self.norm_layer:
            # x=self.pad(x)
            # x=self.norm(x)
            # x=x[:,:,self.p_len:self.p_len+W,self.p_len:self.p_len+H]
        W,H=self.img_size
        x=x.view(-1,W,H,self.in_chans).permute(0,3,1,2)
        x=self.unproj(x)
        # if self.norm_layer:
        #     x=self.pad(x)
        #     x=self.norm(x)
        #     x=x[:,:,self.p_len:self.p_len+W,self.p_len:self.p_len+H]
        x=self.unpooling(x)
        return x


class Transformer_RSTB_floor_wi_classify_res(nn.Module):
    def __init__(self, reslution, patch_size, dim=96, in_c=3, out_c=3, heads=2, window_size=4, b1_depth=2, b2_depth=4,
                 b3_depth=2,drop_path=0.,attn_drop=0.,mlp_ratio=4,expansion=4):
        super(Transformer_RSTB_floor_wi_classify_res, self).__init__()
        self.block0size = (int(reslution / patch_size), int(reslution / patch_size))
        self.block1size = (int(reslution / patch_size ), int(reslution / patch_size ))
        self.block2size = (int(reslution / patch_size / 2), int(reslution / patch_size / 2))
        self.block3size = (int(reslution / patch_size / 4), int(reslution / patch_size / 4))

        self.embed = PatchEmbed(in_chans=in_c, embed_dim=dim, patch_size=patch_size)  # B,C,H,W -> B,HW/patch_size,dim
        # self.merge0=PatchMerging(int(reslution / patch_size), dim)
        self.merge2_1 = PatchMerging_dwt(int(reslution / patch_size ), dim,None)
        self.merge3_1 = PatchMerging_dwt(int(reslution / patch_size ), dim,None)  # B,HW/patch_size,dim -> B,HW/patch_size/4,dim*2
        self.merge2_2 = PatchMerging_dwt(int(reslution / patch_size/2), dim,None)

        self.block1_1 = RSTB_floor_preln(dim, self.block1size, depth=b1_depth, num_heads=heads, window_size=window_size,drop_path=drop_path,attn_drop=attn_drop,mlp_ratio=mlp_ratio)
        self.block1_2 = RSTB_floor_preln(dim, self.block2size, depth=b1_depth, num_heads=heads, window_size=window_size,drop_path=drop_path,attn_drop=attn_drop,mlp_ratio=mlp_ratio)
        self.block1_3 = RSTB_floor_preln(dim, self.block3size, depth=b1_depth, num_heads=heads, window_size=window_size,drop_path=drop_path,attn_drop=attn_drop,mlp_ratio=mlp_ratio)
        self.block2_1 = RSTB_floor_preln_res_qkv(dim*3, self.block1size, depth=b1_depth, num_heads=heads, window_size=window_size,drop_path=drop_path,attn_drop=attn_drop,mlp_ratio=mlp_ratio,expansion=expansion)
        self.block2_2 = RSTB_floor_preln_res_qkv(dim*3, self.block2size, depth=b1_depth, num_heads=heads, window_size=window_size,drop_path=drop_path,attn_drop=attn_drop,mlp_ratio=mlp_ratio,expansion=expansion)
        self.block3_1 = RSTB_floor_preln_res_qkv(dim*3, self.block1size, depth=b1_depth, num_heads=heads, window_size=window_size,drop_path=drop_path,attn_drop=attn_drop,mlp_ratio=mlp_ratio,expansion=expansion)

        self.unmerge2_1 = UnPatchMerging_dwt(self.block2size, dim)
        self.unmerge3_1 = UnPatchMerging_dwt(self.block2size, dim)
        self.unmerge2_2 = UnPatchMerging_dwt(self.block3size, dim)


        self.depatchembed_mode3 = DePatchEmbed(int(reslution / patch_size), patch_size=patch_size, in_chans=dim ,
                                         out_chan=out_c, norm_layer=True)
        self.depatchembed_mode2 = DePatchEmbed(int(reslution / patch_size), patch_size=patch_size, in_chans=dim ,
                                     out_chan=out_c, norm_layer=True)
        self.depatchembed_mode1 = DePatchEmbed(int(reslution / patch_size), patch_size=patch_size, in_chans=dim,
                                               out_chan=out_c, norm_layer=True)

    def get_feature_easy(self,x):
        y_0 = self.embed(x)
        # y_0 = self.merge0(y)
        y_1_1 = self.block1_1(y_0, self.block1size)
        result=self.depatchembed_mode1(y_1_1)
        return result
    def get_feature_mid(self,x):
        y_0 = self.embed(x)
        y_1_1 = self.block1_1(y_0, self.block1size)
        y_1_1_low, y_1_1_high = self.merge2_1(y_1_1)
        y_1_2 = self.block1_2(y_1_1_low, self.block2size)
        y_2_1_high = self.block2_1(y_1_1_high, self.block2size)
        y_2_1 = self.unmerge2_1(y_1_2, y_2_1_high)
        result=self.depatchembed_mode2(y_2_1)
        return result
    def forward(self, x,clist):
        result=torch.zeros_like(x)#.cuda()
        y_0 = self.embed(x)
        # y_0 = self.merge0(y)
        y_1_1 = self.block1_1(y_0, self.block1size)
        if len(clist[0])!=0:
            result[clist[0]] = x[clist[0]] + self.depatchembed_mode1(y_1_1[clist[0]])
            # y = x + self.depatchembed_mode1(y_1_1)
        if len(clist[1])!=0:
            y_1_1_low,y_1_1_high=self.merge2_1(y_1_1[clist[1]])
            y_1_2 = self.block1_2(y_1_1_low, self.block2size)
            y_2_1_high = self.block2_1(y_1_1_high, self.block2size)
            y_2_1 = self.unmerge2_1(y_1_2,y_2_1_high)
            result[clist[1]] = x[clist[1]] + self.depatchembed_mode2(y_2_1)
        if len(clist[2]) != 0:
            y_1_1_low, y_1_1_high = self.merge3_1(y_1_1[clist[2]])
            y_1_2 = self.block1_2(y_1_1_low, self.block2size)
            y_1_2_low, y_1_2_high = self.merge2_2(y_1_2)
            y_1_3 = self.block1_3(y_1_2_low, self.block3size)
            y_2_1_high = self.block2_1(y_1_1_high, self.block2size)
            y_2_2_high = self.block2_2(y_1_2_high, self.block3size)
            y_2_2=self.unmerge2_2(y_1_3,y_2_2_high)
            y_3_1_high = self.block3_1(y_2_1_high, self.block2size)
            y_3_1 = self.unmerge3_1(y_2_2,y_3_1_high)
            result[clist[2]] = x[clist[2]] + self.depatchembed_mode3(y_3_1)
        return result
