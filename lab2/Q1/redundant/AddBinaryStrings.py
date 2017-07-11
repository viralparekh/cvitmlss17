# coding: utf-8
# =============================================================================
# Make a simple RNN learn binray addition 
# Binary string pairs and the sum is generated for a given #numBits
# ============================================================================
# for ML Summer School 2017 at IIIT - HYD
# Authors -seq2seq lab mentors
# Do not share this code or the associated exercises anywhere
# we might be using the same code/ exercies for our future schools/ events
# ==============================================================================


## minesh - !!!! The hidden layer size is set as 10, when its too less like 2 or too high like 200 it fails to converge

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

random.seed( 10 )

def getSample(stringLength, testFlag):
	#takes stringlength as input 
	#returns a sample for the network - an input sequence - x and its target -y
	#x is a T*2 array, T is the length of the string and 2 since we take one bit each from each string
	
	lowerBound=pow(2,stringLength-1)+1
	upperBound=pow(2,stringLength)

	#print ('upper and lower bounds are')
	#print(lowerBound)
	#print(upperBound)
	num1=random.randint(lowerBound,upperBound)
	num2=random.randint(lowerBound,upperBound)

	num3=num1+num2
	num3Binary=(bin(num3)[2:])

	num1Binary=(bin(num1)[2:])

	num2Binary=(bin(num2)[2:])

	if testFlag==1:
		print('input numbers and their sum  are', num1, ' ', num2, ' ', num3)
		print ('binary strings are', num1Binary, ' ' , num2Binary, ' ' , num3Binary)
	len_num1= (len(num1Binary))

	len_num2= (len(num2Binary))
	len_num3= (len(num3Binary))

	# since num3 will be the largest, we pad  other numbers with zeros to that num3_len
	num1Binary= ('0'*(len(num3Binary)-len(num1Binary))+num1Binary)
	num2Binary= ('0'*(len(num3Binary)-len(num2Binary))+num2Binary)


	# forming the input sequence
	# the input at first timestep is the least significant bits of the two input binary strings
	# x will be then a len_num3 ( or T ) * 2 array
	x=np.zeros((len_num3,2),dtype=np.float32)
	for i in range(0, len_num3):
		x[i,0]=num1Binary[len_num3-1-i] # note that MSB of the binray string should be the last input along the time axis
		x[i,1]=num2Binary[len_num3-1-i]
	# target vector is the sum in binary
	# convert binary string in <string> to a numpy 1D array
	#https://stackoverflow.com/questions/29091869/convert-bitstring-string-of-1-and-0s-to-numpy-array
	y=np.array(map(int, num3Binary[::-1]))
	#print (x)
	#print (y)
	return x,y 


#######################
## MODEL DEFINITION
#############################
class Adder (nn.Module):
	def __init__(self, inputDim, hiddenDim, outputDim):
		super(Adder, self).__init__()
		self.inputDim=inputDim
		self.hiddenDim=hiddenDim
		self.outputDim=outputDim
		self.lstm=nn.LSTM(inputDim, hiddenDim )
		self.outputLayer=nn.Linear(hiddenDim, outputDim)
		self.sigmoid=nn.Sigmoid()
	def forward(self, x ):
		#size of x is T x B x featDim
		#B=1 is dummy batch dimension added, because pytorch mandates it
		#if you want B as first dimension of x then specift batchFirst=True when LSTM is initalized
		#T,D  = x.size(0), x.size(1)
		#batch is a must 
		lstmOut,_ =self.lstm(x ) #x has two  dimensions  seqLen *batch* FeatDim=2
		T,B,D  = lstmOut.size(0),lstmOut.size(1) , lstmOut.size(2)
		lstmOut = lstmOut.contiguous() #
		lstmOut = lstmOut.view(B*T, D)
		outputLayerActivations=self.outputLayer(lstmOut)
		#reshape actiavtions to T*B*outputlayersize
		#remove the extra dimesnion we have added for B
		outputLayerActivations=outputLayerActivations.view(T,B,-1).squeeze(1)
		outputSigmoid=self.sigmoid(outputLayerActivations)
		return outputSigmoid




############################################
# TRAINING
################################################


featDim=2 # two bits each from each of the String
outputDim=1 # one output node which would output a zero or 1

lstmSize=20

lossFunction = nn.MSELoss()
model =Adder(featDim, lstmSize, outputDim)
print ('model initialized')
#optimizer = optim.SGD(model.parameters(), lr=3e-2, momentum=0.8)
optimizer=optim.Adam(model.parameters(),lr=0.001)

### epochs ##
totalLoss= float("inf")
while totalLoss > 1e-4:
	print(" Avg. Loss for last 500 samples = %lf"%(totalLoss))
	totalLoss=0
	for i in range(0,500):
		
		stringLen=3
		testFlag=0
		x,y=getSample(stringLen, testFlag)

		model.zero_grad()


		x_var=autograd.Variable(torch.from_numpy(x).unsqueeze(1).float()) #convert to torch tensor and variable
		# unsqueeze() is used to add the extra dimension since
		# your input need to be of t*batchsize*featDim; you cant do away with the batch in pytorch
		seqLen=x_var.size(0)
		#print (x_var)
		x_var= x_var.contiguous()
		y_var=autograd.Variable(torch.from_numpy(y).float())
		finalScores = model(x_var)
		#finalScores=finalScores.

		loss=lossFunction(finalScores,y_var)	
		totalLoss+=loss.data[0]
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		
	totalLoss=totalLoss/500
	
####### Testing ###
stringLen=4
testFlag=1
for i in range (0,10):
	x,y=getSample(stringLen,testFlag)
	x_var=autograd.Variable(torch.from_numpy(x).unsqueeze(1).float())
	y_var=autograd.Variable(torch.from_numpy(y).float())
	seqLen=x_var.size(0)
	x_var= x_var.contiguous()
	finalScores = model(x_var).data.t()
	#print(finalScores)
	bits=finalScores.gt(0.5)
	bits=bits[0].numpy()
	#print(bits)	
	#inverting a 1d tensor in tensor is not easy !!
	#https://github.com/pytorch/pytorch/issues/229
	#inv_idx = torch.arange(bits.size(0)-1, -1, -1).long()
	#inv_tensor = bits.index_select(0, inv_idx)
	#bits_inverted= bits[inv_index]
	
	print ('sum predicted by RNN is ',bits[::-1])
	print('##################################################')

	




