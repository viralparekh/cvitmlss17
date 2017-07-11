import torch
import random

random.seed(1)
op=torch.randn(2,3,2)
#print (op.size())
#print (op)
opt=op.transpose(0,1)
#print(opt)
#print opt.max(2).squeeze(2)
_, argmax=opt.max(2)
print(argmax.squeeze(2))
