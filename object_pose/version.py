import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.zeros(2).cuda())