import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange





class multi_head_kron(nn.Module):
    def __init__(self, dim_in, dim_out, l_in, l_out, heads, layer_num = 0):
        super().__init__()
        self.heads = heads
        self.mat1 = nn.Linear(dim_in, heads * dim_out, bias = False)
        self.mat1.weight = nn.Parameter(torch.nn.init.uniform_(torch.randn(heads * dim_in, dim_out), a = -(3**0.5), b = 3**0.5) * ((2 ** 0.5) / (dim_in * (heads ** 0.5)) ** 0.5))
        self.mat2 = nn.Parameter(torch.nn.init.uniform_(torch.randn(heads,l_in, l_out), a = -(3**0.5), b = 3**0.5) * ((2 ** 0.5) / (l_in * (heads ** 0.5)) ** 0.5))
        self.activation = nn.ReLU()
        self.bias = nn.Parameter(torch.zeros(l_out, dim_out))
        # self.bn = nn.BatchNorm1d(l_out)
        self.layer_num = layer_num

    def forward(self, x):
        print(f'incoming var at layer {self.layer_num}: {torch.var(x)}')
        x = self.mat1(x)
        x = rearrange(x, 'b l (h d) -> b h l d', h = self.heads)
        x = torch.matmul(self.mat2, x)
        x = torch.sum(x, dim = 1)
        x = x + self.bias
        # x = self.bn(x)
        print(f'pre activation var at layer {self.layer_num}: {torch.var(x)}')
        x = self.activation(x)
        print(f'outgoing var at layer {self.layer_num}:  {torch.var(x)}')
        return x

        
x = torch.randn(3, 100, 100) * 2**0.5

model = multi_head_kron(100, 100, 100, 100, 4)

print(torch.var(x))

y = model(x)

print(torch.var(y))