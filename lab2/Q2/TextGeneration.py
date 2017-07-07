# coding: utf-8
# =============================================================================
#  Char-RNN in seq2seq manner  
#  The seq2seq style this code follows is similar to the one in the Karpathy's original code
# ============================================================================
# for ML Summer School 2017 at IIIT - HYD
# Authors -seq2seq lab mentors
# Do not share this code or the associated exercises anywhere
# we might be using the same code/ exercies for our future schools/ events
# ==============================================================================


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

use_cuda = torch.cuda.is_available()
###################################################
# Chunking and Vectorizing the text corpus
##################################################

text = open('tinyshakesepare.txt').read().lower()
print('corpus length:', len(text))


chars = sorted(list(set(text)))
#print(chars)
#print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#print('char is',indices_char[10])


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
#y= np.zeros((len(sentences),maxlen, 1), dtype=np.uint8)
for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		X[i, t, char_indices[char]] = 1.0 

for i, sentence in enumerate(next_chars):
	for t, char in enumerate(sentence):
		y[i, t, char_indices[char]] = 1.0 
		#y[i, t,0] = 1


print ('vectorization complete')




featDim=len(chars)
batchSize=50
totalSequences=len(sentences)





############################################################
#    MODEL DEFINITION 
##########################################################


# minesh -  TODO provision to have more layers ?

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
		self.softmax = nn.Softmax()
		
	#	self.hidden=self.init_hidden()	
	#def init_hidden(self):
	#	 return (autograd.Variable(torch.zeros(self.numLayers*self.numDirections, self.batchSize, self.hiddenDim)),
    #             autograd.Variable(torch.zeros(self.numLayers*self.numDirections, self.batchSize, self.hiddenDim)))
 	

	
	def forward(self, x ):
		B,T,D  = x.size(0), x.size(1), x.size(2)
		lstmOut,_ =self.lstm(x ) #x has three dimensions batchSize* seqLen * FeatDim
		lstmOut = lstmOut.contiguous()
		lstmOut = lstmOut.view(B*T, -1)
		outputLayerActivations=self.outputLayer(lstmOut)
		outputSoftmax=self.softmax(outputLayerActivations)
		if use_cuda:
			outputSoftmax=outputSoftmax.cuda()
		return outputSoftmax
		#if use_cuda:
		#	outputLayerActivations=outputLayerActivations.cuda()
		#return outputLayerActivations

####################################################################
# TRAINING
###################################################################




lstmSize=512
numLstmLayers=1 #how many rnn/lstm layers of above size need to be stacked
numDirections=1 # unidirectional =1 , biDirectional=2

lossFunction = nn.MSELoss()
model = RNNnet( featDim, lstmSize, featDim, numLstmLayers, numDirections,batchSize)
if use_cuda:
	model=model.cuda()
optimizer = optim.RMSprop(model.parameters(), lr=0.01)


seed_string="you are"

# minesh - TODO - shuffle X before every iteration
print ( 'training begins')
### epochs ##
for epoch in range(1):
	step=0
	for i in range(0, totalSequences - maxlen+1, batchSize):# take chunks of size=batchSize in sequential order from X 
		step=step+1
		model.zero_grad()
		#model.hidden = model.init_hidden()
		currentBatchInput=autograd.Variable(torch.from_numpy(X[i:i+batchSize, :, :]).float()) #convert to torch tensor and variable
		if use_cuda:
			currentBatchInput=currentBatchInput.cuda()
		currentBatchInput = currentBatchInput.contiguous()
		#print(currentBatchInput.size())
		currentBatchTarget=autograd.Variable(torch.from_numpy(y[i:i+batchSize, :, :]).float())
		if use_cuda:
			currentBatchTarget=currentBatchTarget.cuda()

		finalScores = model(currentBatchInput)
		finalScores=finalScores.view(batchSize,maxlen,featDim)
		#_, argMaxAtAllTimesteps=finalScores.max(2)
		#print('argmax is', argMaxAtAllTimesteps[0,10,:])

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
					#print('size of activation vector is', finalScores[i,j,:].size())
					#totalLoss = lossFunction(finalScores[i,j,:],currentBatchTarget[i,j,:])
					#print ('outputs and targets are')
					#print (finalScores[i,j,:])
					#print(currentBatchTarget[i,j,:])
					totalLoss = lossFunction(finalScores[i,j,:],currentBatchTarget[i,j,:])
				else:
					totalLoss += lossFunction(finalScores[i,j,:],currentBatchTarget[i,j,:])

		
			
		totalLoss=totalLoss/(maxlen*batchSize)
		#print ('loss is')
		#print (totalLoss)
		
		if step%2==0:
			
			seed_string="you are"
			
			print ("seed string -->", seed_string)
			print ('The generated text is')
			sys.stdout.write(seed_string),
			
			for k in range(0,40):
				x=np.zeros((1, len(seed_string), len(chars)))
				x=autograd.Variable(torch.from_numpy(x).float())
				for t, char in enumerate(seed_string):
					x.data[0, t, char_indices[char]] = 1.
					#print ('type of x', type(x))
					#x=autograd.Variable(torch.from_numpy(x).float())
				scores = model(x)
				#print(scores.size())
				_, argMaxAtAllTimesteps=scores.max(1)
				#print (argMaxAtAllTimesteps.size())
				next_index=argMaxAtAllTimesteps.data[len(seed_string)-1,0]
				#print ('NEXT INDEX IS',next_index)
				#print(scores[0,:])
				#firstIndex=argMaxAtAllTimesteps.data[0,0]
				#firstChar=indices_char[firstIndex]
				#print('######################################first char is',firstChar)

				next_char = indices_char[next_index]

				seed_string = seed_string + next_char
				sys.stdout.write(next_char)
			sys.stdout.flush()

		

		optimizer.zero_grad()
		totalLoss.backward()
		optimizer.step()
		
		
	
	





