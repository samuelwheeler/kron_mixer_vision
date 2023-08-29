import torch
torch.cuda.device(1)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
