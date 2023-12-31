import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class multi_head_kron(nn.Module):
    def __init__(self, dim_in, dim_out, l_in, l_out, heads, layer_num = 0):
        super().__init__()
        self.heads = heads
        self.mat1 = nn.Linear(dim_in, heads * dim_out, bias = False)
        self.mat1.weight = nn.Parameter(torch.nn.init.uniform_(torch.randn( heads * dim_out, dim_in), a = -(3**0.5), b = 3**0.5) * ((2 ** 0.5) / (dim_in * (heads ** 0.5)) ** 0.5))
        self.mat2 = nn.Parameter(torch.nn.init.uniform_(torch.randn(heads, l_out, l_in), a = -(3**0.5), b = 3**0.5) * ((2 ** 0.5) / (l_in * (heads ** 0.5)) ** 0.5))
        # self.activation = nn.ReLU()
        self.bias = nn.Parameter(torch.zeros(l_out, dim_out))
        # self.ln = nn.LayerNorm(dim_out)
        self.layer_num = layer_num

    def forward(self, x):
        x = self.mat1(x)
        x = rearrange(x, 'b l (h d) -> b h l d', h = self.heads)
        x = torch.matmul(self.mat2, x)
        x = torch.sum(x, dim = 1)
        x = x + self.bias
        # x = self.ln(x)
        # x = self.activation(x)
        return x

        


class KronMixer(nn.Module):
    def __init__(self, *, patch_size, num_classes, model_dim, mlp_dim_scale, depth, heads, channels = 3):
        super().__init__()

        patch_height, patch_width = patch_size
        patch_dim = channels * patch_height * patch_width
        num_patches = (32 // patch_height) * (32 // patch_width)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width))
                
        self.layers = nn.ModuleList([])
        for i in range(depth):
                self.layers.append(nn.Sequential(
                    nn.LayerNorm(model_dim),
                    multi_head_kron(model_dim, model_dim, num_patches + 1, num_patches + 1, heads, layer_num = i)
                    ))
                self.layers.append(nn.Sequential(
                    nn.LayerNorm(model_dim),
                    FeedForward(model_dim, mlp_dim_scale * model_dim)
                    ))
                
        self.mlp_head = nn.Linear(model_dim, num_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        # self.to_model_dim = nn.Sequential(
        #     nn.Linear(patch_dim, model_dim),
        #     nn.GELU(),
        #     nn.Linear(model_dim, model_dim),
        # )

        #print the number of parameters for each layer module in the model:
        for name, module in self.named_modules():
            print(name, sum(p.numel() for p in module.parameters()))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        # x = self.to_model_dim(x)
        b, l, d = x.shape
        cls_tokens = repeat(self.cls_token, '() l d -> b l d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        for layer in self.layers:
            x = layer(x) + x
        x = x[:, 0]
        x = self.mlp_head(x)
        return x
