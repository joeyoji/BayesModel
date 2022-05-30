#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import tools
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import scipy.special as spsp
import scipy.stats as spst
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')


# In[6]:


class Poisson_Mixture:
    
    def __init__(self,C_t=1,D=1,N=100,seed=None):
        
        self.generate_sample(C_t,D,N,seed)
        
    def generate_sample(self,C_t,D,N,seed=None):
        
        self.C_t = C_t
        self.D = D
        self.N = N
        
        if seed:
            np.random.seed(seed)
            
        self.mr_t = np.random.dirichlet(np.ones(C_t)) #generate true mixing ratio[C_t]
        self.tens_t = np.random.gamma(np.ones((C_t,D)),                                      D*np.ones((C_t,D))) #generate true intensity[C_t,D]
        
        self.y_t = np.random.multinomial(1,self.mr_t,size=N) #generate true component[N,C_t]
        self.X_t = np.random.poisson(np.sum(self.tens_t[np.newaxis,:,:]*                                            self.y_t[:,:,np.newaxis],                                            axis=1)) #generate raw data[N,D]
        
        self.bin_label,self.bin_weight = np.unique(self.X_t,return_counts=True,axis=0)
        self.N_b = np.size(self.bin_weight)
        #label[N_b,D],weight[D]

    def set_model(self,C,cent,shape,scale,seed=None):

        if C==np.size(cent) and C==np.size(shape,axis=0) and C==np.size(scale,axis=0)            and self.D==np.size(shape,axis=1) and self.D==np.size(scale,axis=1):
            self.C = C
            self.cent = cent
            self.shape = shape
            self.scale = scale
            self.set_GS(seed=seed)
            self.set_VI()
            self.set_CGS()
        else:
            raise TypeError('the size of variables are not corresponding.')
    
    def set_GS(self,L=1,seed=None):
        
        if seed:
            np.random.seed(seed)

        self.L = L
        self.hp_pi_gsc = np.ones((L,self.N_b,self.C))/self.C #hi_pi_gsc[L,N_b,C]
        
        #run one cycle
        self.Gibbs_cycle()
        
    def Gibbs_cycle(self):
        
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
            self.mr_gsc[:,c] = remains*np.random.beta(self.hp_cent_gsc[:,c],                                                      np.sum(self.hp_cent_gsc[:,c+1:],axis=1))
            remains-=self.mr_gsc[:,c]
        self.mr_gsc[:,self.C-1] = remains
        
        #set hp_shape_gsc,hp_scale_gsc and tens_gsc
        self.hp_shape_gsc = np.sum(self.y_gsc[:,:,:,np.newaxis]*                              self.bin_label[np.newaxis,:,np.newaxis,:]                              ,axis=1)+self.shape[np.newaxis,:,:] #[L,C,D]
        self.hp_scale_gsc = self.scale[np.newaxis,:,:]/                            (1+np.sum(self.y_gsc,axis=1)[:,:,np.newaxis]*                            self.scale[np.newaxis,:,:]) #[L,C,D]
        self.tens_gsc = np.random.gamma(self.hp_shape_gsc,self.hp_scale_gsc) #[L,C,D]
        
        #calculate hp_pi_gsc
        self.hp_pi_gsc = self.mr_gsc[:,np.newaxis,:]*np.prod((                        self.tens_gsc[:,np.newaxis,:,:]**                        self.bin_label[np.newaxis,:,np.newaxis,:])/
                        spsp.factorial(self.bin_label[np.newaxis,:,np.newaxis,:])/\
                        np.exp(self.tens_gsc[:,np.newaxis,:,:]),axis=3)
    
    def GibbsSampling(self,ITER=500):
        
        self.mr_GS = np.zeros((ITER,self.L,self.C)) #[I,L,C]
        self.tens_GS = np.zeros((ITER,self.L,self.C,self.D)) #[I,L,C,D]
        for k in tqdm(range(ITER)):
            self.Gibbs_cycle()
            self.mr_GS[k,:,:] = self.mr_gsc
            self.tens_GS[k,:,:,:] = self.tens_gsc
  
    def set_VI(self):
        
        self.hp_cent_vic = self.cent #[C]
        self.hp_shape_vic = self.shape #[C,D]
        self.hp_scale_vic = self.scale #[C,D]
        
        #run one cycle
        self.Variational_cycle()
    
    def Variational_cycle(self):
        
        #calculate hp_pi_vic
        lmd_ex = self.hp_shape_vic*self.hp_scale_vic #[C,D]
        ln_lmd_ex = spsp.digamma(self.hp_shape_vic)+np.log(self.hp_scale_vic) #[C,D]
        ln_pi_ex = spsp.digamma(self.hp_cent_vic)-spsp.digamma(np.sum(self.hp_cent_vic)) #[C]
        self.hp_pi_vic = np.exp(ln_pi_ex)[np.newaxis,:]*np.prod(np.exp(                            self.bin_label[:,np.newaxis,:]*ln_lmd_ex[np.newaxis,:,:]-lmd_ex[np.newaxis,:,:]),axis=2) #[N_b,C]
        self.hp_pi_vic/=np.sum(self.hp_pi_vic,axis=1)[:,np.newaxis]
        
        #calculate hp_shape_vic and hp_scale_vic
        self.hp_shape_vic = np.sum(self.hp_pi_vic[:,:,np.newaxis]*self.bin_label[:,np.newaxis,:]*                                   self.bin_weight[:,np.newaxis,np.newaxis],axis=0)+self.shape
        self.hp_scale_vic = self.scale/(1+np.sum(self.hp_pi_vic[:,:,np.newaxis]*                                                 self.bin_weight[:,np.newaxis,np.newaxis]*self.scale[np.newaxis,:,:],axis=0))
        
        #calculate hp_cent_vic
        self.hp_cent_vic = np.sum(self.hp_pi_vic*self.bin_weight[:,np.newaxis],axis=0)+self.cent
    
    
    def VariationalInference(self,ITER=500):
        
        for k in tqdm(range(ITER)):
            self.Variational_cycle()
    
    def set_CGS(self):
        
        #before>>>
        self.y_cgsc = np.random.multinomial(1,np.ones(self.C)/self.C,size=self.N) #[N,C]
        self.hp_cent_cgsc = np.sum(self.y_cgsc,axis=0)+self.cent #[C]
        self.hp_shape_cgsc = np.sum(self.y_cgsc[:,:,np.newaxis]*self.X_t[:,np.newaxis,:],axis=0)+self.shape #[C,D]
        self.hp_scale_cgsc = (self.scale+np.sum(self.y_cgsc,axis=0)[:,np.newaxis]) #[C,D]
        #<<<
        
        self.Collapsed_cycle()
    
    def Collapsed_cycle(self):
                
        self.hp_cent_cgsc = self.hp_cent_cgsc[np.newaxis,:]-self.y_cgsc #[N,C]
        self.hp_shape_cgsc = self.hp_shape_cgsc[np.newaxis,:,:]-self.y_cgsc[:,:,np.newaxis]*self.X_t[:,np.newaxis,:] #[N,C,D]
        self.hp_scale_cgsc = self.hp_scale_cgsc[np.newaxis,:,:]-self.y_cgsc[:,:,np.newaxis] #[N,C,D]

        nb_p = 1/(1+self.hp_scale_cgsc) #[N,C,D]
        nb_x = spsp.binom(self.X_t[:,np.newaxis,:]+self.hp_shape_cgsc-1,self.hp_shape_cgsc-1)*                    ((1-nb_p)**self.hp_shape_cgsc)*nb_p**self.X_t[:,np.newaxis,:]

        self.hp_pi_cgsc = self.hp_cent_cgsc*np.prod(nb_x,axis=2) #[N,C]
            

        self.y_cgsc = np.zeros((self.N,self.C),dtype=int) #[N,C]
        Wei = np.ones(self.N,dtype=int) #Wei[N]
        
        for c in range(self.C-1):
            prb = self.hp_pi_cgsc[:,c]/np.sum(self.hp_pi_cgsc[:,c:],axis=1)
            prb[np.isnan(prb)] = 0
            self.y_cgsc[:,c] = np.random.binomial(Wei,prb)
            Wei-=self.y_cgsc[:,c]
        self.y_cgsc[:,self.C-1] = Wei
        
        self.hp_cent_cgsc = np.sum(self.y_cgsc,axis=0)+self.cent #[C]
        self.hp_shape_cgsc = np.sum(self.y_cgsc[:,:,np.newaxis]*self.X_t[:,np.newaxis,:],axis=0)+self.shape #[C,D]
        self.hp_scale_cgsc = self.scale+np.sum(self.y_cgsc,axis=0)[:,np.newaxis] #[C,D]
        
    
    def CollapsedGibbsSampling(self,ITER=500):
        
        for k in tqdm(range(ITER)):
            self.Collapsed_cycle()
    
        


# In[7]:


def try_pmm_model(C_t=2,D=3,N=10000,C=2,ITER=10000,seed=None):


    print('generate model...')
    pm = Poisson_Mixture(C_t,D,N,seed)
    print('done.\ntrue parameter :\n',pm.mr_t,'\n',pm.tens_t)

    print('\nset model...')
    pm.set_model(C,10*np.ones(C)+1/(10**np.arange(C)),np.ones((C,D)),D*np.ones((C,D)),seed)
    print('done.')

    print('\ntry Gibbs sampling...')
    pm.GibbsSampling(ITER)
    print('done.\nGibbs sample mean:\n',np.mean(pm.mr_GS,axis=(0,1)),'\n',np.mean(pm.tens_GS,axis=(0,1)))

    print('\ntry variational inference...')
    pm.VariationalInference(ITER)
    print('done.\nhyper parameter :\n',pm.hp_cent_vic,'\n',pm.hp_shape_vic*pm.hp_scale_vic)
    
    print('\ntry collapsed Gibbs sampling...')
    pm.CollapsedGibbsSampling(ITER)
    print('done.\nhyper parameter :\n',pm.hp_cent_cgsc,'\n',pm.hp_shape_cgsc/pm.hp_scale_cgsc)
    
