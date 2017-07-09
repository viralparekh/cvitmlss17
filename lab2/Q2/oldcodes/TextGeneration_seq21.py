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
#input is a sequence of 40 chars and target is the next charaters of each of the sequence 
#for eg: if you maxlen=3 and the text corpus is abcdefghi, your input ---> target pairs will be
# [a,b,c] --> [d], [b,c,d]--->[e]....and so on


maxlen = 40
step = 1 
sentences = []
next_chars = []
for i in range(0, len(text)-500000- (maxlen+2), step):
	sentences.append(text[i: i + maxlen]) #input is from i to maxlen

	next_chars.append(text[i + maxlen]) # output is i+1 to i+1+maxlen

	#if i < 5:
	#	print(text[i: i + maxlen])
	#	print('next char is')
	#	print(text[i + maxlen])
print('no of  sequences:', len(sentences))

print('Vectorization...')

#first axis is the number of sequences = len(sentences)
#second is the seq length = maxlen ( since seqlen is same here for all sequences; which makes batchlearning easier)

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.float)
#y = np.zeros((len(sentences),maxlen, len(chars)), dtype=np.float) # y is also a sequence , or  a seq of 1 hot vectors
y= np.zeros(len(sentences), dtype=np.uint8)
for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		X[i, t, char_indices[char]] = 1.0 

for i, char in enumerate(next_chars):
		#y[i, t, char_indices[char]] = 1.0 
		y[i] = char_indices[char]

print ('vectorization complete')




featDim=len(chars)
batchSize=32
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
		self.lstm=nn.LSTM(inputDim, hiddenDim,2, batch_first=True)
		self.outputLayer=nn.Linear(hiddenDim, outputDim)
		self.softmax = nn.LogSoftmax()
		
	
	def forward(self, x ):
		lstmOut,_ =self.lstm(x ) #x has three dimensions batchSize* seqLen * FeatDim
		B,T,D  = lstmOut.size(0), lstmOut.size(1), lstmOut.size(2)

		# we need the lstmoutput only fron the last time step (t-1)
		
		#print('lstmout sizes before and after reshape')
		#print(lstmOut.size())
		
		lstmOutLastTimeStep=lstmOut[:,T-1,:].squeeze(1)
		#print(lstmOutLastTimeStep.size())
		outputLayerActivations=self.outputLayer(lstmOutLastTimeStep)
		#print(outputLayerActivations.size())
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




lstmSize=256
numLstmLayers=1 #how many rnn/lstm layers of above size need to be stacked
numDirections=1 # unidirectional =1 , biDirectional=2

lossFunction = nn.NLLLoss()
model = RNNnet( featDim, lstmSize, featDim, numLstmLayers, numDirections,batchSize)
if use_cuda:
	model=model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)


seed_string="you are"

# minesh - TODO - shuffle X before every iteration
print ( 'training begins')
### epochs ##
for epoch in range(4):
	step=0
	for i in range(0, totalSequences - maxlen+1, batchSize):# take chunks of size=batchSize in sequential order from X 
		step=step+1
		model.zero_grad()
		#model.hidden = model.init_hidden()
		currentBatchInput=autograd.Variable(torch.from_numpy(X[i:i+batchSize, :, :]).float()) #convert to torch tensor and variable
		#print('inutdim is')
		#print(currentBatchInput.size())
		if use_cuda:
			currentBatchInput=currentBatchInput.cuda()
		currentBatchInput = currentBatchInput.contiguous()
		#print(currentBatchInput.size())
		currentBatchTarget=autograd.Variable(torch.from_numpy(y[i:i+batchSize]).long())
		#currentBatchTarget=currentBatchTarget.view(batchSize*maxlen, -1)
		if use_cuda:
			currentBatchTarget=currentBatchTarget.cuda()

		finalScores = model(currentBatchInput)
		#print('sizes are')
		#print(finalScores.size())
		#print(currentBatchTarget.size())
		loss=lossFunction(finalScores, currentBatchTarget)		
		
		#print ('loss is',loss)
		#print (totalLoss)
		
		if step%10==0:
			print ('loss is',loss)
			seed_string="you ar"
			
			print ("seed string -->", seed_string)
			print ('The generated text is')
			sys.stdout.write(seed_string),
			
			for k in range(0,100):
				x=np.zeros((1, len(seed_string), len(chars)))
				x=autograd.Variable(torch.from_numpy(x).float())
				for t, char in enumerate(seed_string):
					x.data[0, t, char_indices[char]] = 1.
					#print ('type of x', type(x))
					#x=autograd.Variable(torch.from_numpy(x).float())
				scores = model(x)
				scores=scores.squeeze(0)
				#print(scores.size())
				_, argMaxAtAllTimesteps=scores.max(0)
				#print (argMaxAtAllTimesteps.size())
				next_index=argMaxAtAllTimesteps
				#print ('NEXT INDEX IS',next_index)
				#print(scores[0,:])
				#print('\n###############################################')
				#firstIndex=argMaxAtAllTimesteps.data[0,0]
				#firstChar=indices_char[firstIndex]
				#print('######################################first char is',firstChar)

				next_char = indices_char[next_index.data[0]]

				seed_string = seed_string + next_char
				sys.stdout.write(next_char)
			sys.stdout.flush()
			print('\n###############################################')
		
	
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		
