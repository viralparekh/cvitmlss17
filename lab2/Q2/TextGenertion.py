# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 15:22:27 2017
 
@author: Modification of https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb by Michael Klachko
 
Changes:
    - added batch support
    - added multi-GPU support
    - minor changes to train code
    - removed Unicode support (assume input.txt is ASCII)
    - added comments 
 
"""
 
import string
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import time, math
 
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
 
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
 
 
printable = string.printable
 
#Input text is available here: https://sherlock-holm.es/stories/plain-text/cano.txt
text = open('tinyshakesepare.txt', 'r').read()
 
pruned_text = ''
 
for c in text:
    if c in printable and c not in '{}[]&_':
        pruned_text += c
    #else: print c,
 
text = pruned_text        
file_len = len(text)
alphabet = ''.join(sorted(set(text)))
n_chars = len(alphabet)
  
print "\nTraining RNN on Sherlock Holmes novels.\n"      
print "\nFile length: {:d} characters\nUnique characters: {:d}".format(file_len, n_chars)
print "\nUnique characters:", alphabet       
 
def random_chunk():
    start = random.randint(0, file_len - chunk_len)
    end = start + chunk_len + 1
    return text[start:end]
 
def chunk_vector(chunk):
    vector = torch.zeros(len(chunk)).long()
    for i, c in enumerate(chunk):
        vector[i] = alphabet.index(c)  #construct ASCII vector for chunk, one number per character
    return Variable(vector.cuda(), requires_grad=False) 
     
def random_training_sample():    
    chunk = random_chunk()
    inp = chunk_vector(chunk[:-1])
    target = chunk_vector(chunk[1:])
    return inp, target
 
def random_training_batch():
    inputs = []
    targets = []
    #construct list of input vectors (chunk_len):
    for b in range(batch_size):    
        chunk = random_chunk()
        inp = chunk_vector(chunk[:-1])
        target = chunk_vector(chunk[1:])
        inputs.append(inp)
        targets.append(target)
    #construct batches from lists (chunk_len, batch_size):
    #need .view to handle batch_size=1
    #need .contiguous to allow .view later
    inp = torch.cat(inputs, 0).view(batch_size, chunk_len).t().contiguous()
    target = torch.cat(targets, 0).view(batch_size, chunk_len).t().contiguous()
    return inp, target
 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(RNN, self).__init__()
         
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers 
        self.batch_size = batch_size
         
        self.encoder = nn.Embedding(input_size, hidden_size) #first arg is dictionary size
        self.GRU = nn.GRU(hidden_size, hidden_size, n_layers)  #(input_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
         
    def forward(self, input, hidden, batch_size):
        #expand input vector length from single number to hidden_size vector
        #Input: LongTensor (batch, seq_len)
        #Output: (batch, seq_len, hidden_size)
        input = self.encoder(input.view(batch_size, seq_len)) 
        #need to reshape Input to (seq_len, batch, hidden_size)
        input = input.permute(1, 0, 2)
        #Hidden (num_layers * num_directions, batch, hidden_size), num_directions = 2 for BiRNN
        #Output (seq_len, batch, hidden_size * num_directions)
        output, hidden = self.GRU(input, hidden) 
        #output, hidden = self.GRU(input.view(seq_len, batch_size, hidden_size), hidden) 
        #Output becomes (batch, hidden_size * num_directions), seq_len=1 (single char)
        output = self.decoder(output.view(batch_size, hidden_size))  
        #now the output is (batch_size, output_size)
        return output, hidden
     
    def init_hidden(self, batch_size):
        #Hidden (num_layers * num_directions, batch, hidden_size), num_directions = 2 for BiRNN
        return Variable(torch.randn(self.n_layers, batch_size, self.hidden_size).cuda())
 
seq_len = 1        #each character is encoded as a single integer
chunk_len = 128    #number of characters in a single text sample
batch_size = 64   #number of text samples in a batch
n_batches = 200   #size of training dataset (total number of batches)
hidden_size = 256  #width of model
n_layers = 2      #depth of model
LR = 0.005         #learning rate
 
net = RNN(n_chars, hidden_size, n_chars, n_layers).cuda()
#net = RNN(n_chars, hidden_size, n_chars, n_layers)
optim = torch.optim.Adam(net.parameters(), LR)
cost = nn.CrossEntropyLoss().cuda()  
 
print "\nModel parameters:\n"
print "n_batches: {:d}\nbatch_size: {:d}\nchunk_len: {:d}\nhidden_size: {:d}\nn_layers: {:d}\nLR: {:.4f}\n".format(n_batches, batch_size, chunk_len, hidden_size, n_layers, LR)
print "\nRandom chunk of text:\n\n", random_chunk(), '\n'
     
"""
Take input, target pairs of chunks (target is shifted forward by a single character)
convert them into chunk vectors
for each char pair (i, t) in chunk vectors (input, target), create embeddings with dim = hidden_size
feed input char vectors to GRU model, and compute error = output - target
update weights after going through all chars in the chunk
"""
 
def evaluate(prime_str = 'A', predict_len = 100, temp = 0.8, batch_size = 1):
    hidden = net.init_hidden(batch_size) 
    prime_input = chunk_vector(prime_str)
    predicted = prime_str
     
    for i in range(len(prime_str)-1):
        _, hidden = net(prime_input[i], hidden, batch_size)
      
    inp = prime_input[-1]
     
    for i in range(predict_len):
        output, hidden = net(inp, hidden, batch_size)
        output_dist = output.data.view(-1).div(temp).exp()  
        top_i = torch.multinomial(output_dist, 1)[0]
         
        predicted_char = alphabet[top_i]
        predicted +=  predicted_char
        inp = chunk_vector(predicted_char)
 
    return predicted
 
  
start = time.time()
 
training_set = []
 
for i in range(n_batches):
    training_set.append((random_training_batch()))
 
i = 0   
for inp, target in training_set:
    #re-init hidden outputs, zero grads, zero loss:
    hidden = net.init_hidden(batch_size)
    net.zero_grad()
    loss = 0       
    #for each char in a chunk:
    #compute output, error, loss:
    for c, t in zip(inp, target):
        output, hidden = net(c, hidden, batch_size)
        loss += cost(output, t)
    #calculate gradients, update weights:
    loss.backward()
    optim.step()
 
    if i % 100 == 0:
        print "\n\nSample output:\n"
        print evaluate('Wh', 100, 0.8), '\n'
        print('[%s (%d / %d) loss: %.4f]' % (time_since(start), i, n_batches, loss.data[0] / chunk_len))
 
    i += 1
print('i is')
print (i)
