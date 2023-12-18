#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import numpy as np
import scipy.special as spsp
import scipy.stats as spst
from tqdm import tqdm

import datetime
import re
import warnings
# warnings.filterwarnings('ignore')


class Poisson_Mixture_with_Dirichlet_Gamma_prior:
    

    def __init__(self, data):

        '''
        initialize Poisson Mixture Model whose priors are Dirichlet for mixing ratio (Multinomial parameter) and Gamma for intensity (Poisson parameter)

        ===== augments =====

        [1] data[np.2darray] ... the data supposed to be generated from Poisson Mixture ( array-shape (N, D) )
        
            where N is the number of data and D is the dimension of data.
        '''

        if len(data.shape)!=2:
            raise ValueError('data should be 2-dimensional array.')
        
        self.X = data
        self.bin_label, self.bin_weight = np.unique(self.X, return_counts=True, axis=0)
        self.N_b = np.size(self.bin_weight)
        self.N = data.shape[0]
        self.D = data.shape[1]
    
        
    def set_model(self, C, cent=None, shape=None, scale=None, L=1, seed=None):
        
        '''
        initialize Gibbs sample.
        
        ===== arugments =====
        
        [1] C[int] ... the number of model components, which differs from the number of true components
        
        [2] cent[np.1darray or None] ... the hyper parameter of Dirichlet prior distribution ( array-shape (C,) )
        
        [3] shape[np.2darray or None] ... one of the hyper parameter of Gamma prior distribution ( array-shape (C,D) )
        
        [4] scale[np.2darray or None] ... the other of the hyper parameter of Gamma prior distribution ( array-shape (C,D) )

        [5] L[int] ... the number of parallel Gibbs series
        
        [6] seed[int or None] ... random seed to set initial parameter
        '''

        if type(cent)==type(None):
            cent = 10*(np.ones(C)+1/(10**np.arange(C)))
        if type(shape)==type(None):
            shape = np.ones((C,self.D))
        if type(scale)==type(None):
            scale = self.D*np.ones((C,self.D))
        
        if C==np.size(cent) and C==np.size(shape,axis=0) and C==np.size(scale,axis=0) and self.D==np.size(shape,axis=1) and self.D==np.size(scale,axis=1):
            self.C = C
            self.cent = cent
            self.shape = shape
            self.scale = scale

            if seed:
                np.random.seed(seed)

            self.L = L
            #set hp_pi_gsc
            self.hp_pi_gsc = np.ones((self.L,self.N_b,self.C))/self.C #hi_pi_gsc[L,N_b,C]
            
            #run one cycle
            self.Gibbs_cycle()
        else:
            raise TypeError('the size of variables are not corresponding.')
       
    
    def Gibbs_cycle(self):
        
        '''
        run one Gibbs sampling cycle
        '''
        
        #set y_gsc
        self.y_gsc = np.zeros((self.L,self.N_b,self.C),dtype=int) #[L,N_b,C]
        Wei = np.repeat([np.copy(self.bin_weight),],self.L,axis=0) #Wei[L,N_b]
        for c in range(self.C-1):
            prb = self.hp_pi_gsc[:,:,c]/np.sum(self.hp_pi_gsc[:,:,c:],axis=2)
            prb[np.isnan(prb)] = 0
            self.y_gsc[:,:,c] = np.random.binomial(Wei,prb)
            Wei-=self.y_gsc[:,:,c]
        self.y_gsc[:,:,self.C-1] = Wei
        
        #set hp_cent_gsc & mr_gsc
        self.mr_gsc = np.zeros((self.L,self.C)) #[L,C]
        self.hp_cent_gsc = np.sum(self.y_gsc,axis=1)+self.cent #[L,C]
        remains = np.ones(self.L)
        for c in range(self.C-1):
            self.mr_gsc[:,c] = remains*np.random.beta(self.hp_cent_gsc[:,c],np.sum(self.hp_cent_gsc[:,c+1:],axis=1))
            remains-=self.mr_gsc[:,c]
        self.mr_gsc[:,self.C-1] = remains
        
        #set hp_shape_gsc,hp_scale_gsc and tens_gsc
        self.hp_shape_gsc = np.sum(self.y_gsc[:,:,:,np.newaxis]*self.bin_label[np.newaxis,:,np.newaxis,:],axis=1)+self.shape[np.newaxis,:,:] #[L,C,D]
        self.hp_scale_gsc = self.scale[np.newaxis,:,:]/(1+np.sum(self.y_gsc,axis=1)[:,:,np.newaxis]*self.scale[np.newaxis,:,:]) #[L,C,D]
        self.tens_gsc = np.random.gamma(self.hp_shape_gsc,self.hp_scale_gsc) #[L,C,D]
        
        #calculate hp_pi_gsc
        self.hp_pi_gsc = (self.mr_gsc[:,np.newaxis,:]*np.prod((self.tens_gsc[:,np.newaxis,:,:]**self.bin_label[np.newaxis,:,np.newaxis,:])/
                        spsp.factorial(self.bin_label[np.newaxis,:,np.newaxis,:])/\
                        np.exp(self.tens_gsc[:,np.newaxis,:,:]),axis=3)) #[L,N_b,C]
    
    
    def GibbsSampling(self, ITER=500, burnin=None, seed=None):
        
        '''
        
        run Gibbs sampling
        
        ===== arguments =====
        
        ITER[int] ... the number of times to turn GS for sampling
        
        burnin[int or None] ... the number of times to turn GS for stabilizing sample
        
        need_elbo[bool] ... whether return ELBO or not
        
        seed[int or None] ... random seed for Gibbs sampling
        
        '''
        
        if seed:
            np.random.seed(seed)
            
        if burnin==None:
            burnin = self.N
            
        print('burn-in...')
        for k in tqdm(range(burnin)):
            self.Gibbs_cycle()
        
        print('sampling...')
        self.mr_GS = np.zeros((ITER,self.L,self.C)) #[I,L,C]
        self.tens_GS = np.zeros((ITER,self.L,self.C,self.D)) #[I,L,C,D]
        for k in tqdm(range(ITER)):
            self.Gibbs_cycle()
            self.mr_GS[k,:,:] = self.mr_gsc
            self.tens_GS[k,:,:,:] = self.tens_gsc

        return self.mr_GS, self.tens_GS


def try_pmm_model(C=2,ITER=10000,seed=42):

    np.random.seed(seed)
    data = np.random.poisson(3,size=(10000, 1))

    print('generate model...')
    pm = Poisson_Mixture_with_Dirichlet_Gamma_prior(data)
    print('done.')

    print('\nset model...')
    pm.set_model(C, seed=seed)
    print('done.')

    print('\ntry Gibbs sampling...')
    mr_gs, tens_gs = pm.GibbsSampling(ITER, seed=seed)
    print('done.\nGibbs sample mean:\n',np.mean(mr_gs,axis=(0,1)),'\n',np.mean(tens_gs,axis=(0,1)))


if __name__ == '__main__':

    try_pmm_model()