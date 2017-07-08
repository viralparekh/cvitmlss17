from __future__ import print_function
import numpy as np
from time import sleep
import random
import sys 
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


op=torch.FloatTensor([0.6, 0.2, 0.2] ).unsqueeze(0)
target=torch.LongTensor([0])
print (target.size())
op=autograd.Variable(op)
target=autograd.Variable(target)
lossFunction = nn.CrossEntropyLoss()
l=lossFunction(op,target)

