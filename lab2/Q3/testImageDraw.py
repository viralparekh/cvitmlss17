from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import codecs,subprocess,os,sys,glob
import random

random.seed()

# ref : https://stackoverflow.com/questions/15857117/python-pil-text-to-image-and-fonts

fontsList=glob.glob('/OCRData/minesh.mathew/Englishfonts/English_fonts/googleFonts/'+'*.ttf')
vocabFile=codecs.open('/OCRData2/minesh.mathew/oxford_dataset/sct/mnt/ramdisk/max/90kDICT32px/lexicon.txt','r')
words = vocabFile.read().split()

for i,fontName in enumerate(fontsList):
	print (fontName)
	#myFont=random.sample(fontsList,1)[0]
	#TODO - sample the integer directly 
	#fontSizeOptions={'16','20','24','28','30','32','36','38'}
	myFontSize=30
	#print (myFontSize)
	myText="eshmathew"
#myFontSize=46
	myFont = ImageFont.truetype(fontName,int(myFontSize))
	myTextSize=myFont.getsize(myText)
	


