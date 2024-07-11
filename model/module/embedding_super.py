import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import to_2tuple
import numpy as np

class PatchembedSuper(nn.Module): #all changed to 32
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=256, scale=False, num_patches=200):
        super(PatchembedSuper, self).__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # patch_num = to_2tuple(147)
        # num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        num_patches = num_patches
        #calculate num of patches
        self.img_size = img_size
        self.patch_size = patch_size
        # self.patch_num = patch_num
        self.num_patches = num_patches
        ## for hsi in_chans=img_size*img_size*args.in_chans
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.proj = nn.Conv2d(in_chans, num_patches, kernel_size=patch_size, stride=patch_size)

        # self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # using conv instead of linear layer (hybird)
        self.super_embed_dim = embed_dim
        self.scale = scale

    # sampled_
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None
        # self.linear_layer = torch.nn.Linear(self.patch_num, self.sample_embed_dim)
        # self.patch_embed = None
        # self.patch_to_embedding = nn.Linear(7 * 7 * 3, self.sample_embed_dim)

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight = self.proj.weight[:sample_embed_dim, ...]
        self.sampled_bias = self.proj.bias[:self.sample_embed_dim, ...]
        if self.scale:
            self.sampled_scale = self.super_embed_dim / sample_embed_dim
    def forward(self, x):
        # print(x.shape)
        # B, C, H, W = x.shape #batch size,channel,h,w=resolution of images
        B, C, _ = x.shape
        # self.patch_embed = P
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = torch.unsqueeze(x, dim=0)
        x = x.unsqueeze(2).permute(0, 3, 2, 1)
        x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1,2)
        # x = self.linear_layer(x)
 
        # print(x.shape)
        # x = self.patch_to_embedding(x)
        if self.scale:
            return x * self.sampled_scale
        return x
    def calc_sampled_param_num(self):
        return  self.sampled_weight.numel() + self.sampled_bias.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops
