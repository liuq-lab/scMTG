import tensorflow as tf
from .model import Decoder2, Generator, Discriminator, BaseFullyConnectedNet
import numpy as np
import pandas as pd
from .util import Sequential_sampler
from .loss import NB
import dateutil.tz
import datetime
import sys
import copy
import os
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class scMTG(object):
    """Markov Transition Generative model for time-series single cell analysis.
    """
    def __init__(self, params, timestamp=None, random_seed=None):
        super(scMTG, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)

        #initilize the shared autoencoder (encoder + decoder)
        self.encoder = BaseFullyConnectedNet(input_dim=params['e_dim'],z_dim=params['z_dim'],output_dim = params['z_dim'], 
                                        model_name='e_net', nb_units=params['e_units'], last_relu=False)
        self.decoder = Decoder2(input_dim=params['z_dim'],z_dim=params['z_dim'],output_dim = params['e_dim'],
                               model_name='d_net', nb_units=params['d_units'], last_relu=False)

        #initilize the T-1 Markov generators and T-1 discriminators
        self.generators = [Generator(input_dim=params['noise_dim']+params['z_dim'],z_dim=params['z_dim'],
                                     output_dim=params['z_dim'],model_name='g_net_%d'%i,
                                     nb_units=params['gen_units'], concat_every_fcl=False, last_relu=True) 
                           for i in range(params['nb_time']-1)]
        self.discriminators = [Discriminator(input_dim=params['z_dim'],z_dim=params['z_dim'],output_dim = 1,
                                             model_name='d_net_%d'%i, nb_units=params['dis_units'], last_relu=False) 
                               for i in range(params['nb_time']-1)]
        lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(params['lr'], decay_steps=100000, decay_rate=0.9, staircase=True)
        lr_schedule2 = tf.keras.optimizers.schedules.ExponentialDecay(params['lr']/10.0, decay_steps=100000, decay_rate=0.9, staircase=True)
        self.ae_optimizer = tf.keras.optimizers.Adam(lr_schedule1, beta_1=0.5, beta_2=0.9)
        self.e_optimizer = tf.keras.optimizers.Adam(lr_schedule2, beta_1=0.5, beta_2=0.9)
        self.g_optimizer = tf.keras.optimizers.Adam(lr_schedule1, beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(lr_schedule1, beta_1=0.5, beta_2=0.9)

        self.initialize_nets()

        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime('%Y%m%d_%H%M%S')
            
        self.best_path = "{}/{}/best_model/".format(
            params['output_dir'], self.timestamp)

        if self.params['save_model'] and not os.path.exists(self.best_path):
            os.makedirs(self.best_path)
        
        self.save_dir = "{}/{}".format(
            params['output_dir'], self.timestamp)

        if self.params['save_res'] and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)   

        self.ckpt = tf.train.Checkpoint(encoder = self.encoder,
                                   decoder = self.decoder,
                                   generators = self.generators,
                                   discriminators = self.discriminators,
                                   ae_optimizer = self.ae_optimizer,
                                   e_optimizer = self.e_optimizer,
                                   g_optimizer = self.g_optimizer,
                                   d_optimizer = self.d_optimizer)

    def get_config(self):
        return {
                "params": self.params,
        }

    def initialize_nets(self, print_summary = True):
        """Initialize all the networks in CausalEGM."""

        self.encoder(np.zeros((1, self.params['e_dim'])))
        self.decoder(np.zeros((1, self.params['z_dim'])), 1.0)
        [self.generators[i](np.zeros((1, self.params['z_dim']+self.params['noise_dim']))) 
            for i in range(self.params['nb_time']-1)]
        [self.discriminators[i](np.zeros((1, self.params['z_dim']))) 
            for i in range(self.params['nb_time']-1)]
        if print_summary:
            print(self.encoder.summary())
            print(self.decoder.summary())
            print([self.generators[i].summary() for i in range(self.params['nb_time']-1)])
            print([self.discriminators[i].summary() for i in range(self.params['nb_time']-1)])

    @tf.function
    def train_ae_step(self, data_series):
        """train shared AE.
        """  
        with tf.GradientTape(persistent=True) as tape:
            embed_series = tf.map_fn(lambda item:self.encoder(item), data_series)
            
            disps, means = [], []
            for i in range(len(embed_series)):
                disp, mean = self.decoder(embed_series[i])
                disps.append(disp)
                means.append(mean)
            zinb = NB(theta=tf.concat(disps, axis=0), debug=False)
            loss_rec = zinb.loss(tf.reshape(data_series, [-1, data_series.shape[-1]]), tf.concat(means, axis=0), mean=True)
    
        # Calculate the gradients
        gradients = tape.gradient(loss_rec, self.encoder.trainable_variables+self.decoder.trainable_variables)
        # Apply the gradients to the optimizer
        self.ae_optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables+self.decoder.trainable_variables))
        return loss_rec

    @tf.function
    def train_gen_step(self, data_series):
        """train generators.
        """  
        noise = tf.random.normal(shape=(self.params['nb_time']-1,data_series.shape[1],self.params['noise_dim']),mean=0.,stddev=self.params['sd'])
        with tf.GradientTape(persistent=True) as gen_tape:
            embed_series = tf.map_fn(lambda item:self.encoder(item) ,data_series)

            #data from time point 1,2,...,T-1
            data_previous = tf.concat([embed_series[:-1],noise],axis=-1)

            #generated data for time point 2,...,T
            data_gen = tf.TensorArray(tf.float32, size=data_previous.shape[0])
            for i in range(data_previous.shape[0]):
                data_gen = data_gen.write(i, self.generators[i](data_previous[i]))
            data_gen=data_gen.stack()

            dz_gen = tf.TensorArray(tf.float32, size=data_gen.shape[0])
            for i in range(data_gen.shape[0]):
                dz_gen = dz_gen.write(i, self.discriminators[i](data_gen[i]))
            dz_gen=dz_gen.stack()

            loss_g = -tf.reduce_mean(dz_gen)
            loss_td = tf.reduce_mean((data_gen-embed_series[:-1])**2)
            loss_g_total = loss_g + self.params['beta']*loss_td

        # Calculate the gradients
        g_gradients = gen_tape.gradient(loss_g_total, sum([item.trainable_variables for item in self.generators], []))
        self.g_optimizer.apply_gradients(zip(g_gradients, sum([item.trainable_variables for item in self.generators], [])))
        
        e_gradients = gen_tape.gradient(loss_g_total, self.encoder.trainable_variables)
        self.e_optimizer.apply_gradients(zip(e_gradients, self.encoder.trainable_variables))
        return loss_g, loss_td, loss_g_total

    @tf.function
    def train_disc_step(self, data_series):
        """train discriminators.
        """  
        epsilon_z = tf.random.uniform(shape=(self.params['nb_time']-1,1,1),minval=0., maxval=1.)
        noise = tf.random.normal(shape=(self.params['nb_time']-1,data_series.shape[1],self.params['noise_dim']),mean=0.,stddev=self.params['sd'])
        with tf.GradientTape(persistent=True) as disc_tape:
            embed_series = tf.map_fn(lambda item:self.encoder(item) ,data_series)
            data_previous = tf.concat([embed_series[:-1],noise],axis=-1)

            data_gen = tf.TensorArray(tf.float32, size=self.params['nb_time']-1)
            for i in range(self.params['nb_time']-1):
                data_gen = data_gen.write(i, self.generators[i](data_previous[i]))
            data_gen=data_gen.stack()
            
            data_true = embed_series[1:]

            dz_gen = tf.TensorArray(tf.float32, size=self.params['nb_time']-1)
            for i in range(self.params['nb_time']-1):
                dz_gen = dz_gen.write(i, self.discriminators[i](data_gen[i]))
            dz_gen=dz_gen.stack()

            dz_true = tf.TensorArray(tf.float32, size=self.params['nb_time']-1)
            for i in range(self.params['nb_time']-1):
                dz_true = dz_true.write(i, self.discriminators[i](data_true[i]))
            dz_true = dz_true.stack()
            loss_d = tf.reduce_mean(dz_gen)-tf.reduce_mean(dz_true)
            
            #gradient penalty for z
            data_hat = epsilon_z*data_gen+(1-epsilon_z)*data_true

            dz_hat = tf.TensorArray(tf.float32, size=self.params['nb_time']-1)
            for i in range(self.params['nb_time']-1):
                dz_hat = dz_hat.write(i, self.discriminators[i](data_hat[i]))
            dz_hat=dz_hat.stack()
            
            grads = tf.gradients(dz_hat,data_hat)[0]
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
            loss_gp = tf.reduce_mean(tf.square(grad_norms - 1))

            loss_d_total = loss_d + self.params['alpha']*loss_gp

        # Calculate the gradients
        d_gradients = disc_tape.gradient(loss_d_total, sum([item.trainable_variables for item in self.discriminators], []))
        self.d_optimizer.apply_gradients(zip(d_gradients, sum([item.trainable_variables for item in self.discriminators], [])))
        return loss_d, loss_gp, loss_d_total

    def train(self, data, weights, batch_size=32, n_iter=100000, batches_per_verbose=500, verbose=1):
        if self.params['save_res']:
            f_params = open('{}/params.txt'.format(self.save_dir),'w')
            f_params.write(str(self.params))
            f_params.close()
        self.data_sampler = Sequential_sampler(data=data, weights=weights, batch_size=batch_size)
        
        best_loss = float('inf')
        best_ckpt = None
        best_batch_idx = 0
        loss_pd = []
        for batch_idx in range(n_iter+1):
            batch_data_series = self.data_sampler.next_batch()

            #train autoencoders
            loss_rec = self.train_ae_step(batch_data_series)

            #control the update frequency of autoencoder vs GAN
            if batch_idx % self.params['ae_gan_freq'] == 0:
                #update G once and update D multiple times for WGAN-GP
                for _ in range(self.params['g_d_freq']):
                    batch_data_series = self.data_sampler.next_batch()
                    loss_d, loss_gp, loss_d_total = self.train_disc_step(batch_data_series)
                batch_data_series = self.data_sampler.next_batch()
                loss_g, loss_td, loss_g_total = self.train_gen_step(batch_data_series)
            
            loss_total = loss_rec + loss_g_total + loss_d_total
            loss_pd.append([batch_idx, loss_total.numpy(), loss_rec.numpy(), loss_gp.numpy(), loss_d.numpy(), loss_g.numpy(), loss_td.numpy()])
            if loss_total < best_loss:
                best_loss = loss_total
                best_ckpt = self.ckpt
                best_batch_idx = batch_idx
                self.ckpt.save('{}/{}/best_model'.format(self.best_path, batch_idx))
            if batch_idx - best_batch_idx >= 10000 and batch_idx >= 50000:
                print("Early stop at {} and best at {}!".format(batch_idx, best_batch_idx))
                break
                
            if batch_idx % batches_per_verbose == 0:
                loss_contents = '''Iteration [%d] : loss_t [%.4f], loss_rec [%.4f], loss_gp [%.4f], loss_d [%.4f], loss_g [%.4f] loss_td [%.4f]''' \
                %(batch_idx, loss_total, loss_rec, loss_gp, loss_d, loss_g, loss_td)
                if verbose:
                    print(loss_contents)
        loss_pd = pd.DataFrame(loss_pd, columns=['batch_idx', 'loss_total', 'loss_rec', 'loss_gp', 'loss_d', 'loss_g', 'loss_td'])
        loss_pd.to_csv(self.save_dir+'/loss_pd.csv', sep = '\t', index = False)
#         self.ckpt.save('{}/best_model_at_{}'.format(self.best_path, batch_idx))
        self.ckpt.restore(tf.train.latest_checkpoint('{}/best_model_at_{}'.format(self.best_path, best_batch_idx)))
        self.evaluate(self.data_sampler.load_all(), 'best')
    
    def evaluate(self, data_series, batch_idx):
        embed_series = [self.encoder.predict(item) for item in data_series]
        data_previous = [np.concatenate([data, np.random.normal(0.,self.params['sd'],size=(data.shape[0],self.params['noise_dim']))],axis=1) for data in embed_series[:-1]] #contain T-1 time points
        data_gen_z = [self.generators[i].predict(data) for i,data in enumerate(data_previous)]
        data_gen_org = [self.decoder.predict(item)[-1] for item in data_gen_z]
        
        np.savez('{}/data_embed_at_{}.npz'.format(self.save_dir, batch_idx),embed_series)
        np.savez('{}/data_gen_at_{}.npz'.format(self.save_dir, batch_idx),data_gen_z)
        np.savez('{}/data_gen_org_at_{}.npz'.format(self.save_dir, batch_idx),data_gen_org)

    def thresholding_(self, trans_mtx):
        trans_mtx2 = trans_mtx / np.sum(trans_mtx, axis=1, keepdims=True)
        trans_mtx2[np.isnan(trans_mtx2)] = 0.0

        trans_mtx0 = trans_mtx2[np.where(np.sum(trans_mtx2, axis=1)>0)[0]]
        thresh = min(trans_mtx0[i].max() for i in range(trans_mtx0.shape[0]))
#         print(thresh)

        trans_mtx3 = trans_mtx2.copy()
        trans_mtx3[trans_mtx3<thresh] = 0.0
        trans_mtx3 = trans_mtx3 / np.sum(trans_mtx3, axis=1, keepdims=True)
        trans_mtx3[np.isnan(trans_mtx3)] = 0.0
        return trans_mtx3
    
    def compute_trans_mat(self, times=1, n_noise=1000, n_chunk=1000, random_seed=1, thresholding=True, save_mtx=False):
        data_series = self.data_sampler.load_all()
        embed_series = [self.encoder.predict(item) for item in data_series]
        
        tf.random.set_seed(random_seed)
        noises = tf.random.normal(shape=(n_noise, self.params['noise_dim']), mean=0.0, stddev=self.params['sd'])

        embed_data = embed_series[times]
        trans_mtx = []
        const = -0.5 / (self.params['sd'] ** 2)
        for embed_data0 in embed_data:
            embed_gen1 = self.generators[times].predict(tf.concat([tf.tile(tf.reshape(embed_data0,(1,-1)),[n_noise,1]),noises], axis=-1))
            trans2 = []
            for i in range(n_noise//n_chunk):
                trans2.append(tf.reduce_mean(tf.math.exp(const*tf.reduce_sum((tf.reshape(tf.tile(embed_series[times+1], [1,n_chunk]), [embed_series[times+1].shape[0],-1,embed_series[times+1].shape[1]])-embed_gen1[i*n_chunk:(i+1)*n_chunk])**2,axis=-1)),axis=-1))
            trans_mtx.append(tf.reduce_mean(trans2, axis=0))
        trans_mtx = np.array(trans_mtx)
        
        if thresholding:
            trans_mtx = self.thresholding_(trans_mtx)
            
        if save_mtx:
            np.savez('{}/trans_mtx_{}.npz'.format(self.save_dir, times),t1=trans_mtx)
        return trans_mtx

    
    
