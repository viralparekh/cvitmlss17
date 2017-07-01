# =============================================================================
# Use RNNs to add two strings of binary (1/0) numbers.
#
# ==============================================================================
from __future__ import print_function, division

import numpy as np


class RandomBinaryStrings(object):
  """
  Generates random training / validation / test sets, where each sample consists
  of: two input binary strings of length self.STR_LEN, and an output binary
  string, also of length self.STR_LEN, which is the sum of the two input strings.
  Note, the "carry" bit (if any) of the last addition is discarded.
  """
  def __init__(self,str_len=10,n_train=100,n_val=100,n_test=100,rand_seed=42):
    """
    STR_LEN: length of the binary strings
    N_{TRAIN,VAL,TEST}: number of strings in training/val/test splits
    MAKE_EXCLUSIVE: (default False), if true, the dataset-splits are exclusive,
                    i.e., a given binary string belongs to only one of the splits
                    If False, a repitition will happen with low probability
                    (~ 1/2**STR_LEN)
    RAND_SEED: random seed for the random number generator (for reproducibility)
    """
    self.str_len = str_len
    self.prng = np.random.RandomState(rand_seed) # random-number generator
    # split the range into train/val/test:
    assert n_train < int(np.ceil(0.8*(2**str_len))),\
      "number of bit-strings in the training set exceed 80%% of the total"\
      + "possible bit-strings of length %d"%str_len
    # get the train / val / test splits:
    rand_strings = self._get_exclusive_binary_strings(n_train+n_val+n_test)
    train = rand_strings[:,:n_train]
    val = rand_strings[:,n_train:n_train+n_val]
    test = rand_strings[:,n_train+n_val:]
    self.data_splits = {'train':train,'val':val,'test':test}

  def _get_exclusive_binary_strings(self,n):
    """
    Returns a [STR_LEN,N] uint8 tensor of N unique random binary strings.
    Note: A more "efficient" implemented is not used to avoid dealing with
          "huge" numbers.
    """
    # make sure that n strings can be generated:
    assert n <= 2**self.str_len,\
      'The number of requested unique binary strings (=%d) exceeds the maximum'\
      + 'possible with %d bits (=%d).'%(n,self.str_len,2**self.str_len)
    # generate:
    n_samples = 0; s = np.zeros([self.str_len,n],dtype=np.uint8)
    while n_samples < n:
      b = (self.prng.rand(self.str_len)>0.5).astype(np.uint8)
      if not np.any(np.all(s[:,:n_samples]==b[:,None],axis=0)):
        s[:,n_samples] = b
        n_samples += 1
    return s

  def _add(self,a,b):
    """
    Adds two binary tensors, of size [LEN,N] each. Final carry is discarded.
    Implements "full-adder".
    """
    l,n = a.shape
    c = np.zeros(n,dtype=a.dtype)
    s = np.zeros_like(a)
    a_xor_b = np.bitwise_xor(a,b)
    a_and_b = np.bitwise_and(a,b)
    for i in xrange(l):
      s[i,:] = np.bitwise_xor(a_xor_b[i,:],c)
      c = np.bitwise_or(a_and_b[i,:], np.bitwise_and(c,a_xor_b[i,:]))
    return s

  def get_batch(self,bsz,data_split):
    """
    Returns batches of size: [self.STR_LEN,BSZ,3], where the in the
    last-dimension, the first two are inputs, and the third is the sum of the
    two inputs.
    """
    try:
      src = self.data_splits[data_split]
    except KeyError,e:
      raise ValueError('Data split %s unknown.'%data_split)
    n_src = len(src)
    ins = self.prng.randint(0,n_src,size=2*bsz)
    ins = src[:,ins].reshape(-1,bsz,2)
    # add the "ins" to get the output:
    outs = np.expand_dims(self._add(ins[...,0],ins[...,1]),axis=2)
    batch = np.concatenate([ins,outs],axis=2)
    return batch
