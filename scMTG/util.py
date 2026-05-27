import numpy as np
import os
import math
import pandas as pd


class Base_sampler(object):
    """Base data sampler.

    Parameters
    ----------
    x
        List or Numpy.ndarray bject denoting the treatment with length N or shape (N, 1) or (N, ). 
    y
        List or Numpy.ndarray bject denoting the outcome with length N or shape (N, 1) or (N, ). 
    batch_size
        Int object denoting the batch size for mini-batch training. Default: ``32``.

    Examples
    --------
    >>> from CausalEGM import Base_sampler
    >>> import numpy as np
    >>> x = np.random.normal(size=(2000,))
    >>> y = np.random.normal(size=(2000,))
    >>> v = np.random.normal(size=(2000,100))
    >>> ds = Base_sampler(x=x,y=y,v=v)
    >>> batch = ds.next_batch() # get a batch of data
    >>> data = ds.load_all() # get all data as a triplet
    """
    def __init__(self, x, y, batch_size=32, normalize=False, random_seed=123):
        assert x.shape[0]==y.shape[0]
        np.random.seed(random_seed)
        self.batch_size = batch_size
        self.sample_size = x.shape[0]
        self.full_index = np.arange(self.sample_size)
        np.random.shuffle(self.full_index)
        self.idx_gen = self.create_idx_generator(sample_size=self.sample_size)
        self.data_x = np.array(x, dtype='float32')
        self.data_y = np.array(y, dtype='float32')
        
    def create_idx_generator(self, sample_size, random_seed=123):
        while True:
            for step in range(math.ceil(sample_size/self.batch_size)):
                if (step+1)*self.batch_size <= sample_size:
                    yield self.full_index[step*self.batch_size:(step+1)*self.batch_size]
                else:
                    yield np.hstack([self.full_index[step*self.batch_size:],
                                    self.full_index[:((step+1)*self.batch_size-sample_size)]])
                    np.random.shuffle(self.full_index)

    def next_batch(self):
        indx = next(self.idx_gen)
        return self.data_x[indx,:], self.data_y[indx,:]
    
    def load_all(self):
        return self.data_x, self.data_y

class Sequential_sampler(object):
    def __init__(self, data, batch_size=32, random_seed=123):
        np.random.seed(random_seed)
        self.data = [np.array(item, dtype='float32') for item in data]
        self.nb_time = len(self.data)
        self.batch_size = batch_size
        self.sample_sizes = [item.shape[0] for item in self.data]
        self.full_indexes = [np.arange(item) for item in self.sample_sizes]
        [np.random.shuffle(item) for item in self.full_indexes]
        self.idx_gens = [self.create_idx_generator(sample_size=item, time_idx=i) for i,item in enumerate(self.sample_sizes)]
        
    def create_idx_generator(self, sample_size, time_idx, random_seed=123):
        while True:
            for step in range(math.ceil(sample_size/self.batch_size)):
                if (step+1)*self.batch_size <= sample_size:
                    yield self.full_indexes[time_idx][step*self.batch_size:(step+1)*self.batch_size]
                    if (step+1)*self.batch_size == sample_size:
                        np.random.shuffle(self.full_indexes[time_idx])
                else:
                    yield np.hstack([self.full_indexes[time_idx][step*self.batch_size:],
                                    self.full_indexes[time_idx][:((step+1)*self.batch_size-sample_size)]])
                    np.random.shuffle(self.full_indexes[time_idx])

    def next_batch(self):
        indexes = [next(item) for item in self.idx_gens]
        return np.stack([item[indexes[i],:] for i,item in enumerate(self.data)])
    
    def load_all(self):
        return self.data
    