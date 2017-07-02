import numpy as np
nBits=16
inputDim=2 # one bit from each string
outputDim=1 # one output node which would output a 0 or 1
batchSize=64 # 64 samples in a batch
dtype = torch.cuda.FloatTensor #to be used if the code is running on a GPU
np.random.seed(0)
#a=np.array([12,13])
#b=np.array([10,20])
#c=np.array([a[0],b[0]])

#samples to be generated on the fly - 64  pairs of numbers and the sums


