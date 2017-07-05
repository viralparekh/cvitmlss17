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
import torchvision.transforms as transforms
from warpctc_pytorch import CTCLoss
random.seed(0)

fontsList=glob.glob('/OCRData/minesh.mathew/Englishfonts/English_fonts/googleFonts/'+'*.ttf')
vocabFile=codecs.open('/OCRData2/minesh.mathew/oxford_dataset/sct/mnt/ramdisk/max/90kDICT32px/lexicon.txt','r')
words = vocabFile.read().split()
vocabSize=len(words)
fontSizeOptions={'16','20','24','28','30','32','36','38'}
batchSize=10
alphabet='0123456789abcdefghijklmnopqrstuvwxyz-'
dict={}
for i, char in enumerate(alphabet):
	dict[char] = i + 1


def StrtoLabels(text):
	global dict
	text = [dict[char.lower()] for char in text]
	#print (text)
	length=len(text)
	return text, length
#StrtoLabels("0-1")


def image2tensor(im):

    """
	input - a PIL Image
	output - a torch tensor of the shape  H*W
	ref : https://stackoverflow.com/questions/13550376/pil-image-to-array-numpy-array-to-array-python
    """
    (width, height) = im.size
    greyscale_map = list(im.getdata())
    greyscale_map = np.array(greyscale_map)
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
		fontName=random.sample(fontsList,1)[0]
		fontSize=random.sample(fontSizeOptions,1)[0]
		imageFont = ImageFont.truetype(fontName,int(fontSize))
		textSize=imageFont.getsize(wordText)
		img=Image.new("L", textSize,(255))
		draw = ImageDraw.Draw(img)
		draw.text((0, 0),wordText,(0),font=imageFont)
		img=img.resize((100,32), Image.ANTIALIAS)
		imgTensor=image2tensor(img).unsqueeze(0) # at 0 a new dimension is added
		#print (imgTensor.size())
		#print (imgArray.shape)
		wordImages.append(imgTensor)

		labelSeq,l=StrtoLabels(wordText)
		labelSequences+=labelSeq
		labelSeqLengths.append(l)
	batchImageTensor=torch.cat(wordImages,0) #now all the image tensors are combined ( we  did the unsqueeze eariler for this cat)	
	batchImageTensor=torch.transpose(batchImageTensor,1,2)
	labelSequencesTensor=torch.IntTensor(labelSequences)
	labelSeqLengthsTensor=torch.IntTensor(labelSeqLengths)
	print(batchImageTensor.size())
	#print (labelSequencesTensor)
	return batchImageTensor, labelSequencesTensor, labelSeqLengthsTensor
		


#####################################################
# MODEL DEFINITION PART
####################################################


# minesh TODO split blstm into a separate class ?

class rnnocr (nn.Module):
	def __init__(self, inputDim, hiddenDim, outputDim,  numLayers, numDirections,batchSize):
		super(rnnocr, self).__init__()
		self.inputDim=inputDim
		self.hiddenDim=hiddenDim
		self.outputDim=outputDim
		self.numLayers=numLayers
		self.numDirections=numDirections
		self.batchSize=batchSize

		self.blstm1=nn.LSTM(inputDim, hiddenDim, bidirectional=True, batch_first=True) # first blstm layer takes the image features as inputs
		self.blstm2=nn.LSTM(hiddenDim, hiddenDim, bidirectional=True, batch_first=True) # here input is output of linear layer 1 
		
		self.linearLayer1=nn.Linear(hiddenDim*2, hiddenDim) # the embedding layer between the two blstm layers
		self.linearLayer2=nn.Linear(hiddenDim*2, outputDim) # linear layer at the output
		
		self.softmax = nn.LogSoftmax()
		
		#self.hidden1=self.init_hidden()
		#self.hidden2=self.init_hidden() 
	def init_hidden(self):
		return (autograd.Variable(torch.zeros(self.numLayers*self.numDirections, self.batchSize, self.hiddenDim)),
                autograd.Variable(torch.zeros(self.numLayers*self.numDirections, self.batchSize, self.hiddenDim)))
	def forward(self, x ):
		B,T,D  = x.size(0), x.size(1), x.size(2)
		#lstmOut1, self.hidden1=self.blstm1(x, self.hidden1 ) #x has three dimensions batchSize* seqLen * FeatDim
		lstmOut1, _  =self.blstm1(x ) #x has three dimensions batchSize* seqLen * FeatDim
		B,T,D  = lstmOut1.size(0), lstmOut1.size(1), lstmOut1.size(2)
		lstmOut1=lstmOut1.contiguous()
		embedding=self.linearLayer1(lstmOut1.view(B*T,D))

		input2blstm2=embedding.view(B,T,-1)
		

		#lstmOut2, self.hidden2=self.blstm2(input2blstm2)
		lstmOut2, _ = self.blstm2(input2blstm2)
		B,T,D  = lstmOut2.size(0), lstmOut2.size(1), lstmOut2.size(2)
		lstmOut2=lstmOut2.contiguous()
		outputLayerActivations=self.linearLayer2(lstmOut2.view(B*T,D))
		#outputSoftmax=F.log_softmax(outputLayerActivations)
		return outputLayerActivations.view(B,T,-1).transpose(0,1) # transpose since ctc expects the probabilites to be in t x b x nclasses format






###########################################
# TRAINING
##################################################
"""
a batch of words are sequentially fetched from the vocabulary
one epoch runs until all the words in the vocabulary are seen once
then the word list is shuffled and above process is repeated
"""

nClasses= len(alphabet)
model = rnnocr(32,100,nClasses,1,2,batchSize)
criterion = CTCLoss()
#nClasses= len(alphabet)
optimizer = optim.Adam(model.parameters(), lr=0.01,
                           betas=(0.5, 0.999))





random.shuffle(words)
for i in range (0,vocabSize-batchSize+1,batchSize):
	model.zero_grad()
	model.hidden = model.init_hidden()
	batchOfWords=words[i:i+batchSize]
	#print (len(batchOfWords))
	images,labelSeqs,labelSeqlens =GetBatch(batchOfWords)
	images=autograd.Variable(images)
	images=images.contiguous()
	labelSeqs=autograd.Variable(labelSeqs)
	labelSeqlens=autograd.Variable(labelSeqlens)
	outputs=model(images)
	#print (outputs.size())
	outputs=outputs.contiguous()
	outputsSize=autograd.Variable(torch.IntTensor([outputs.size(0)] * batchSize))
	cost = criterion(outputs, labelSeqs, outputsSize, labelSeqlens) / batchSize
	print (cost)
	optimizer.zero_grad()
	cost.backward()
	optimizer.step()
	






