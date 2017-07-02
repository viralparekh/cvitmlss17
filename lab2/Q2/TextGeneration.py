# coding: utf-8
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


###### vectorize the text #####

text = open('tinyshakesepare.txt').read().lower()
print('corpus length:', len(text))


chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))



# split the corpus into sequences of length=maxlen
#input is a sequence of 40 chars and target is also a sequence of 40 chars shifted by one position
#for eg: if you maxlen=3 and the text corpus is abcdefghi, your input ---> target pairs will be
# [a,b,c] --> [b,c,d], [b,c,d]--->[c,d,e]....and so on


maxlen = 40
step = 1 
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen+1, step):
    sentences.append(text[i: i + maxlen]) #input is from i to maxlen
    next_chars.append(text[i+1:i +1+ maxlen]) # output is i+1 to i+1+maxlen
print('no of  sequences:', len(sentences))

print('Vectorization...')

#first axis is the number of sequences = len(sentences)
#second is the seq length = maxlen ( since seqlen is same here for all sequences; which makes batchlearning easier)
#third axis is the dimensionality of your vector representation, which is = size of the vocabulary = len(chars)

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.float)
y = np.zeros((len(sentences),maxlen, len(chars)), dtype=np.float) # y is also a sequence , or  a seq of 1 hot vectors

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1.0 

for i, sentence in enumerate(next_chars):
    for t, char in enumerate(sentence):
        y[i, t, char_indices[char]] = 1 


print ('vectorization complete')




featDim=len(chars)
batchSize=20
totalSequences=len(sentences)
#X = autograd.Variable(torch.randn((11,maxlen, featDim)))
#y=  autograd.Variable(torch.zeros((11,maxlen, featDim)))
#print (X.type())
#print(y.type())
#y=y.long()

#X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.uint8)
#y = np.zeros((len(sentences),maxlen, len(chars)), dtype=np.uint8) # y is also a sequence , or  a seq of 1 hot vectors





#########  Data is prepared now we must begin building the model ############


#lstm to outputlayer > reshaping?
#what if you have more than one lstm layer?

class RNNnet (nn.Module):
        def __init__(self, inputDim, hiddenDim, outputDim,  numLayers, numDirections,batchSize):
                super(RNNnet, self).__init__()
                self.inputDim=inputDim
                self.hiddenDim=hiddenDim
                self.outputDim=outputDim
                self.numLayers=numLayers
                self.numDirections=numDirections
		self.batchSize=batchSize

                self.lstm=nn.LSTM(inputDim, hiddenDim, batch_first=True)
                self.outputLayer=nn.Linear(hiddenDim, outputDim)
		self.softmax = nn.LogSoftmax()
		self.hidden=self.init_hidden()
        def init_hidden(self):
                #self.h_0=autograd.Variable(torch.randn(self.numLayers*self.numDirections, 1, self.hiddenDim))
                #self.c_0=autograd.Variable(torch.randn(self.numLayers*self.numDirections, 1, self.hiddenDim))
		return (autograd.Variable(torch.zeros(self.numLayers*self.numDirections, self.batchSize, self.hiddenDim)),
                autograd.Variable(torch.zeros(self.numLayers*self.numDirections, self.batchSize, self.hiddenDim)))

        def forward(self, x ):
		B,T,D  = x.size(0), x.size(1), x.size(2)
		#print (B)
		#print (T)
		#print (D)
                lstmOut, self.hidden=self.lstm(x, self.hidden ) #x has three dimensions batchSize* seqLen * FeatDim
		#print ('lstmoutputsize is')
		#print (lstmOut.size())
		lstmOut = lstmOut.contiguous()
		lstmOut = lstmOut.view(B*T, -1)
                outputLayerActivations=self.outputLayer(lstmOut)
		outputSoftmax=F.log_softmax(outputLayerActivations)
                return outputSoftmax




##### Training Code #########
lstmSize=512
numLstmLayers=1 #how many rnn/lstm layers of above size need to be stacked
numDirections=1 # unidirectional =1 , biDirectional=2

lossFunction = nn.NLLLoss()
model = RNNnet( featDim, lstmSize, featDim, numLstmLayers, numDirections,batchSize)
optimizer = optim.SGD(model.parameters(), lr=0.1)


#how to give batches?
print ( 'training begins')
### epochs ##
for epoch in range(1):
	for i in range(0, totalSequences - maxlen+1, batchSize):# take chunks of size=batchSize in sequential order from X 
		model.zero_grad()
		model.hidden = model.init_hidden()
	
		currentBatchInput=autograd.Variable(torch.from_numpy(X[i:i+batchSize, :, :]).float()) #convert to torch tensor and variable
		currentBatchInput = currentBatchInput.contiguous()
		#print(currentBatchInput.size())
		currentBatchTarget=autograd.Variable(torch.from_numpy(y[i:i+batchSize, :, :]).long())
		finalScores = model(currentBatchInput)
		finalScores=finalScores.view(batchSize,maxlen,featDim)
		#print ('size of finalscores')
		#print (finalScores.size())

		#print ('size of target tensor')
		#print (currentBatchTarget.size())

		#lossfunctions we have in torch computes loss between two vectors ( cant handle sequences of such vectors)
		#in our case we need to find the loss at each timestep then add up losses at each timestep to get the total loss for the sequence

		#numInstances=batchSize*maxlen # number of instances is a multiple of seqlen*batchsize, because you have those many instances where you need to compute loss
		totalLossList=[]
		totalLoss = None
		for i in range (0, batchSize):
			for j in range (0, maxlen):
				if totalLoss is None:
					#print (finalScores[i,j,:])
					#print (currentBatchTarget[i,j,:])
					totalLoss = lossFunction(finalScores[i,j,:],currentBatchTarget[i,j,:])

				else:
					totalLoss += lossFunction(finalScores[i,j,:],currentBatchTarget[i,j,:])

		
			
		totalLoss=totalLoss/(maxlen*batchSize)
		print ('loss is')
		print (totalLoss)
		
		totalLoss.backward()
		optimizer.step()
		
		
	
	





