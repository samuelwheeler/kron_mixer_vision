import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class multi_head_kron(nn.Module):
    def __init__(self, dim_in, dim_out, l_in, l_out, heads):
        super().__init__()
        self.heads = heads
        self.mat1 = nn.Linear(dim_in, heads * dim_out, bias = False)
        self.mat1.weight = nn.Parameter(torch.randn(heads * dim_in, dim_out) * (1/dim_in))
        self.mat2 = nn.Parameter(torch.randn(heads, l_in, l_out)) * (1/l_in)
        self.activation = nn.ReLU()
        self.bias = nn.Parameter(torch.zeros(l_out, dim_out))
        self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, x):
        x = self.mat1(x)
        x = rearrange(x, 'b (h d) l -> b h d l', h = self.heads)
        x = torch.matmul(x, self.mat2)
        x = torch.sum(x, dim = 1)
        x = x + self.bias
        x = self.bn(x)
        x = self.activation(x)
        return x

        


class KronMixer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim_l, dim_d, depth, heads, channels = 3):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width))

                
        layers = []
        for _ in range(depth - 1):
            layers.append(multi_head_kron(patch_dim, patch_dim, num_patches, num_patches, heads))
        layers.append(multi_head_kron(patch_dim, dim_l, num_patches, dim_d, heads))

        self.layers = nn.ModuleList(layers)


        self.mlp_head = nn.Linear(dim_l*dim_d, num_classes)
            
        

    def forward(self, img):
        x = self.to_patch_embedding(img)
        
        for layer in self.layers:
            x = layer(x) + x
            # x = self.dropout(x)
        x = rearrange(x, 'b n d -> b (n d)')
        return self.mlp_head(x)