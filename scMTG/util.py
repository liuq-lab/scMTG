import numpy as np
import os
import math
import pandas as pd
import scipy
import anndata as ad
import scib


def _logistic(x, L, k, center=0):
    return L / (1 + np.exp(-k * (x - center)))

def _gen_logistic(p, sup, inf, center, width):
    return inf + _logistic(p, L=sup - inf, k=4.0 / width, center=center)

def beta(p, beta_max=1.7, beta_min=0.3, beta_center=0.25, beta_width=0.5):
    return _gen_logistic(p, beta_max, beta_min, beta_center, beta_width)

def delta(a, delta_max=1.7, delta_min=0.3, delta_center=0.1, delta_width=0.2):
    return _gen_logistic(a, delta_max, delta_min, delta_center, delta_width)

def growth_rate(adata, proliferation_key="proliferation", apoptosis_key="apoptosis", delta_t=1.0, 
                beta_max=1.7, beta_min=0.3, beta_center=0.25, beta_width=0.5, 
                delta_max=1.7, delta_min=0.3, delta_center=0.1, delta_width=0.2):
    birth = beta(adata.obs[proliferation_key].values, 
                 beta_max=beta_max, beta_min=beta_min, beta_center=beta_center, beta_width=beta_width)
    death = delta(adata.obs[apoptosis_key].values, 
                  delta_max=delta_max, delta_min=delta_min, delta_center=delta_center, delta_width=delta_width)
    gr = np.exp((birth - death) * delta_t)
    return gr

class Sequential_sampler(object):
    def __init__(self, data, weights=None, batch_size=32, random_seed=123):
        np.random.seed(random_seed)
        self.data = [np.array(item, dtype='float32') for item in data]
        self.nb_time = len(self.data)
        self.batch_size = batch_size
        self.sample_sizes = [item.shape[0] for item in self.data]
        if weights is None:
            self.weights = [np.ones(item, dtype='float32') / item for item in self.sample_sizes]
        else:
            self.weights = [np.array(item, dtype='float32') / np.sum(item) for item in weights]
        self.idx_gens = [self.create_idx_generator(sample_size=item, time_idx=i) for i,item in enumerate(self.sample_sizes)]
        
    def create_idx_generator(self, sample_size, time_idx, random_seed=123):
        while True:
#             indices = np.random.choice(sample_size, size=self.batch_size, replace=True, p=self.weights[time_idx])
            indices = np.random.choice(sample_size, size=self.batch_size, replace=False, p=self.weights[time_idx])
            yield indices
#             indices = np.random.choice(sample_size, size=3*self.batch_size, replace=False, p=self.weights[time_idx])
#             np.random.shuffle(indices)
#             yield indices[:self.batch_size]

    def next_batch(self):
        indexes = [next(item) for item in self.idx_gens]
        return np.stack([item[indexes[i],:] for i,item in enumerate(self.data)])
    
    def load_all(self):
        return self.data

def interpolate(p0, p1, tmap, interp_frac, size, seed=1):
    p0 = p0.toarray() if scipy.sparse.isspmatrix(p0) else p0
    p1 = p1.toarray() if scipy.sparse.isspmatrix(p1) else p1
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    tmap = np.asarray(tmap, dtype=np.float64)
    if p0.shape[1] != p1.shape[1]:
        raise ValueError("Unable to interpolate. Number of genes do not match")
    if p0.shape[0] != tmap.shape[0] or p1.shape[0] != tmap.shape[1]:
        raise ValueError("Unable to interpolate. Tmap size is {}, expected {}"
                         .format(tmap.shape, (len(p0), len(p1))))
    I = len(p0);
    J = len(p1)
    a = np.power(tmap.sum(axis=0), 1. - interp_frac)
    a[a==0] = np.finfo(float).eps
    p = tmap / a
    p = p.flatten(order='C')
    p = p / p.sum()
    np.random.seed(seed)
    choices = np.random.choice(I * J, p=p, size=size)
    return np.asarray([p0[i // J] * (1 - interp_frac) + p1[i % J] * interp_frac for i in choices], dtype=np.float64)
    
def cal_metrics(gen_data, real_data):
    mmd_value = cal_mmd(gen_data, real_data)
    
    x, y = np.mean(real_data, axis=0), np.mean(gen_data, axis=0)
    pearson_corr, _ = scipy.stats.pearsonr(x, y)
    spearman_corr, _ = scipy.stats.spearmanr(x, y)
    
    lisi_value = cal_lisi(gen_data, real_data)

    return mmd_value, pearson_corr, spearman_corr, lisi_value

def cal_mmd(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    mmd_value = np.mean((delta[:-1] * delta[1:]).sum(1))
    return mmd_value

def cal_lisi(gen_data, real_data):
    adata_scib = ad.AnnData(np.concatenate((gen_data, real_data), axis=0))
    adata_scib.obs['batch'] = pd.Categorical(['gen']*len(gen_data) + ['true']*len(real_data))
    lisi_value = scib.me.ilisi_graph(adata_scib, batch_key="batch", type_="full")
    return lisi_value
  
    
    
    
