from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import codecs,subprocess,os,sys,glob
import random

random.seed()

# ref : https://stackoverflow.com/questions/15857117/python-pil-text-to-image-and-fonts

fontsList=glob.glob('/OCRData/minesh.mathew/Englishfonts/English_fonts/googleFonts/'+'*.ttf')
vocabFile=codecs.open('/OCRData2/minesh.mathew/oxford_dataset/sct/mnt/ramdisk/max/90kDICT32px/lexicon.txt','r')
words = vocabFile.read().split()

myFont=random.sample(fontsList,1)[0]
#TODO - sample the integer directly 
fontSizeOptions={'16','20','24','28','30','32','36','38'}
myFontSize=random.sample(fontSizeOptions,1)[0]
print (myFontSize)
myText="mineshmathewmineshmathew"
#myFontSize=46
myFont = ImageFont.truetype(myFont,int(myFontSize))
myTextSize=myFont.getsize(myText)

#print (myTextSize)
img=Image.new("L", myTextSize,(255))
draw = ImageDraw.Draw(img)
draw.text((0, 0),myText,(0),font=myFont)
img.save("a_test.png")


#numWords=len(words)
#print 'number of words in the vocab= ', numWords

#writeDir=writeDirParent+'0\/'
#for i in range(0,numWords):

#if i%1000==0:
#    textWord=words[i]

