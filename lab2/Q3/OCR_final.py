# =============================================================================
# Use a BRNN + CTC to recognize given word image 
# Network is trained on images rendered using PIL 
# ============================================================================
# for ML Summer School 2017 at IIIT - HYD
# Authors -seq2seq lab mentors
# Do not share this code or the associated exercises anywhere
# we might be using the same code/ exercies for our future schools/ events
# ==============================================================================

from __future__ import print_function
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np
from time import sleep
import random
import sys,codecs,glob 
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from warpctc_pytorch import CTCLoss
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
random.seed(0)

#all word images are resized to a height of 32 pixels
imHeight=32 
"""
image width is also set a fixed size
YES. Though RNNS can handle variable length sequences we resize them to fixed width
This is for the ease of batch learning
And it doesnt seem to affect the performance much atleast in our case

Pytorch provides a packed array API incase you want to have variable length sequences within a batch
see the discussion here
https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120/8

"""
#imWidth=100
imWidth=15
fontsList=glob.glob('/OCRData/minesh.mathew/Englishfonts/English_fonts/googleFonts/'+'*.ttf')
vocabFile=codecs.open('/OCRData2/minesh.mathew/oxford_dataset/sct/mnt/ramdisk/max/90kDICT32px/lexicon.txt','r')
words = vocabFile.read().split()
vocabSize=len(words)
fontSizeOptions={'16','20','24','28','30','32','36','38'}
batchSize=10
alphabet='0123456789abcdefghijklmnopqrstuvwxyz-'
#alphabet="(3)-"
dict={}
for i, char in enumerate(alphabet):
	dict[char] = i + 1




def Str2Labels(text):
	global dict
	text = [dict[char.lower()] for char in text]
	#print (text)
	length=len(text)
	return text, length
#StrtoLabels("0-1")

def Labels2Str(predictedLabelSequences):
	bz=predictedLabelSequences.size(0)
	predictedRawStrings=[]
	predictedStrings=[]
	for i in range(0,bz):
		predictedRawString=""
		predictedLabelSeq=predictedLabelSequences.data[i,:]
		for j in range (0, predictedLabelSeq.size(0)):
			idx=predictedLabelSeq[j]
			if idx==0:
				character="~"
			else:
				character=alphabet[idx-1]

				
			predictedRawString+=character
		predictedRawStrings.append(predictedRawString)
	return predictedRawStrings



def image2tensor(im):

    (width, height) = im.size
    greyscale_map = list(im.getdata())
    greyscale_map = np.array(greyscale_map, dtype = np.uint8)
    greyscale_map=greyscale_map.astype(float)
    greyscale_map = torch.from_numpy(greyscale_map.reshape((height, width))).float()/255.0
    return greyscale_map


def GetBatch ( batchOfWords ):
	"""
	Renders a batch of word images and returns the images along with the corresponding GTs
	Uses PIL to render word images
	font is randomly picked from a set of freely available google fonts
	word is picked from a vocabulary of English words

	"""
	wordImages=[]
	labelSequences=[]
	labelSeqLengths=[]

	for  i,text in enumerate (batchOfWords):
		wordText=text
		#print('text is', text)
		fontName=fontsList[0]
		fontSize='26'
		#fontSize=fontSizeOptions[0]
		fontName=random.sample(fontsList,1)[0]
		fontSize=random.sample(fontSizeOptions,1)[0]
		imageFont = ImageFont.truetype(fontName,int(fontSize))
		textSize=imageFont.getsize(wordText)
		img=Image.new("L", textSize,(255))
		draw = ImageDraw.Draw(img)
		draw.text((0, 0),wordText,(0),font=imageFont)
		img=img.resize((imWidth,imHeight), Image.ANTIALIAS)
		#img.save(text+'.jpeg')

		imgTensor=image2tensor(img)
		imgTensor=imgTensor.unsqueeze(0) # at 0 a new dimenion is added

		wordImages.append(imgTensor)

		labelSeq,l=Str2Labels(wordText)
		labelSequences+=labelSeq
		labelSeqLengths.append(l)
	batchImageTensor=torch.cat(wordImages,0) #now all the image tensors are combined ( we  did the unsqueeze eariler for this cat)	
	batchImageTensor=torch.transpose(batchImageTensor,1,2)
	labelSequencesTensor=torch.IntTensor(labelSequences)
	labelSeqLengthsTensor=torch.IntTensor(labelSeqLengths)
	return batchImageTensor, labelSequencesTensor, labelSeqLengthsTensor
		


#####################################################
# MODEL DEFINITION PART
####################################################


# minesh TODO split blstm into a separate class ?

class rnnocr (nn.Module):
	def __init__(self, inputDim, hiddenDim, outputDim,  numLayers, numDirections):
		super(rnnocr, self).__init__()
		self.inputDim=inputDim
		self.hiddenDim=hiddenDim
		self.outputDim=outputDim
		self.numLayers=numLayers
		self.numDirections=numDirections

		self.blstm1=nn.LSTM(inputDim, hiddenDim,1, bidirectional=False, batch_first=True) # first blstm layer takes the image features as inputs
		
		self.linearLayer2=nn.Linear(hiddenDim, outputDim) # linear layer at the output
		self.softmax = nn.Softmax()
		
	def forward(self, x ):
		B,T,D  = x.size(0), x.size(1), x.size(2)
		lstmOut1, _  =self.blstm1(x ) #x has three dimensions batchSize* seqLen * FeatDim
		B,T,D  = lstmOut1.size(0), lstmOut1.size(1), lstmOut1.size(2)
		lstmOut1=lstmOut1.contiguous()

		

		outputLayerActivations=self.linearLayer2(lstmOut1.view(B*T,D))
		outputSoftMax=self.softmax(outputLayerActivations)
		return outputLayerActivations.view(B,T,-1).transpose(0,1)



###########
# Prepare the synthetic validation data
##############

valWords=['intermittently','hyderabad','golconda','charminar','gachibowli']
valImages, valLabelSeqs, valLabelSeqlens=GetBatch(valWords)
valImages=autograd.Variable(valImages)
valImages=valImages.contiguous()
valLabelSeqs=autograd.Variable(valLabelSeqs)
#print(valLabelSeqs.data)
valLabelSeqlens=autograd.Variable(valLabelSeqlens)




###########################################
# TRAINING
##################################################
"""
a batch of words are sequentially fetched from the vocabulary
one epoch runs until all the words in the vocabulary are seen once
then the word list is shuffled and above process is repeated
"""
nHidden=100
nClasses= len(alphabet)
criterion = CTCLoss()

numLayers=1 # the 2 BLSTM layers defined seprately without using numLayers option for nn.LSTM
numDirections=2 # 2 since we need to use a bidirectional LSTM
model = rnnocr(imHeight,nHidden,nClasses,numLayers,numDirections)

optimizer=optim.Adam(model.parameters(), lr=0.001)



for iter in range (0,200):
	avgTrainCost=0
	random.shuffle(words)

	for i in range (0,vocabSize-batchSize+1,batchSize):
	
		model.zero_grad()
		
		batchOfWords=words[i:i+batchSize]
		images,labelSeqs,labelSeqlens =GetBatch(batchOfWords)
		images=autograd.Variable(images)
		images=images.contiguous()
		labelSeqs=autograd.Variable(labelSeqs)
		labelSeqlens=autograd.Variable(labelSeqlens)
		outputs=model(images)
		outputs=outputs.contiguous()
		outputsSize=autograd.Variable(torch.IntTensor([outputs.size(0)] * batchSize))
		trainCost = criterion(outputs, labelSeqs, outputsSize, labelSeqlens) / batchSize

		avgTrainCost+=trainCost
		if i%50==0:
			avgTrainCost=avgTrainCost/(5000/batchSize)
			#print ('avgTraincost for last 5000 samples is',avgTrainCost)
			avgTrainCost=0
			valOutputs=model(valImages)
			#print (valOutputs.size()) 100 X nvalsamoles x 37
			valOutputs=valOutputs.contiguous()
			valOutputsSize=autograd.Variable(torch.IntTensor([valOutputs.size(0)] * len(valWords)))
			valCost=criterion(valOutputs, valLabelSeqs, valOutputsSize, valLabelSeqlens) / len(valWords)
			print ('validaton Cost is',valCost)


			### get the actual predictions and compute word error ################
			valOutputs_batchFirst=valOutputs.transpose(0,1)
			# second output of max() is the argmax along the requuired dimension
			_, argMaxActivations= valOutputs_batchFirst.max(2)
			#the below tensor each raw is the sequences of labels predicted for each sample in the batch
			predictedSeqLabels=argMaxActivations.squeeze(2) #batchSize * seqLen 
			predictedStrings=Labels2Str(predictedSeqLabels)
			for ii in range(0,5):

				print (predictedStrings[ii])
		
			#	print (predictedSeqLabels[0,:].transpose(0,0))
			#print(valOutputs_batchFirst[0,0,:])
			#print (argMaxActivations[0,:])

		
		optimizer.zero_grad()
		trainCost.backward()
		optimizer.step()
	#iterString=int(iter)
	#torch.save(model.state_dict(), iterString+'.pth')
	






