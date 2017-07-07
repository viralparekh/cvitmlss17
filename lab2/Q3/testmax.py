import torch

t=torch.FloatTensor([[1, 2, 3,-12,-0.05,0], [4, 5, 6,-23,-1.56,0]])
val,indices=t.max(1)
print(val)
print(indices)
