#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import tools
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import numpy as np
import scipy.special as spsp
import scipy.stats as spst
from tqdm.notebook import tqdm

import datetime
import re
import warnings
# warnings.filterwarnings('ignore')


# In[103]:


class Poisson_Mixture:
    
    '''
    
    this class is initialized optimally for args.
    there are three ways to initialize.
    
    [1] (data[np.2darray]) ; one argument
    
        you can give this a np.2darray. given np.2darray is interpreted as data from Poisson mixture.
        
    [2] (C_t[int], D[int], N[int] (, seed[int]) ) ; three(four) arguments

        you can give this triple (quadruple) integers.
        given integers are interpreted as the number of components in true distribution,
        the dimension of data, and the number of data (,and random seed) respectively.
        if you give them, true parameter and data from it are generated randomly.
        you can use seed argument to fix randomness.

    [3] (mr[np.1darray], tens[np.2darray], N[int] (, seed[int]) ) ; thee(four) arguments

        you can give np.1darray, np.2darray, and integer(s).
        first np.1darray is interpreted as mixing ratio, second np.2darray is done as intensity,
        and third integer is done as the number of data (,and fourth integer is done as random seed).
        the size of mr must correspond to the size of 0-axis of tens.
        if you give them, data from designated true parameter is generated randomly.
        you can use seed argument to fix randomness.
    
    '''
    
    
    def __init__(self,*args):
        
        if len(args)==1 and type(args[0])==np.ndarray:
            self.load_data(args[0])
        elif len(args)==3:
            self.set_true_parameter(args[0],args[1])
            self.generate_sample(args[2])
        elif len(args)==4:
            self.set_true_parameter(args[0],args[1],args[3])
            self.generate_sample(args[2],args[3])
        else:
            raise TypeError('wrong args. check docstring.')
                                 

    def load_data(self,data):
        
        if len(data.shape)!=2:
            raise TypeError('the shape of data array must be N x D, where N is the number of data and D is the dimension of data.')
        else:
            
            self.C_t = None
            self.mr_t = None
            self.tens_t = None
            self.is_true_known = False
            
            self.X_t = data
            self.N = np.size(self.X_t,axis=0)
            self.D = np.size(self.X_t,axis=1)
            
            self.bin_X_t()
                    
                
    def bin_X_t(self):
        
        self.bin_label,self.bin_weight = np.unique(self.X_t,return_counts=True,axis=0)
        self.N_b = np.size(self.bin_weight)
            
        
    def set_true_parameter(self,arg1,arg2,seed=None):
        
        if type(arg1)==np.ndarray and type(arg2)==np.ndarray:
            self.set_true_parameter_by_array(arg1,arg2)
        elif type(arg1)==int and type(arg2)==int:
            self.set_true_parameter_by_int(arg1,arg2,seed)
        else:
            raise TypeError('wrong args. check docstring.')
            
            
    def set_true_parameter_by_array(self,mr,tens):
        
        if mr.shape[0]!=tens.shape[0]:
            raise TypeError('wrong args. check docstring.')
        else:
            self.C_t = np.size(mr)
            self.D = np.size(tens,axis=1)
            self.mr_t = mr
            self.tens_t = tens
            self.is_true_known = True
            
            
    def set_true_parameter_by_int(self,C_t,D,seed=None):
        
        if C_t<=0 or D<=0:
            raise TypeError('C_t and D must be greater than 0.')
        
        if seed:
            np.random.seed(seed)
            
        mr = np.random.dirichlet(np.ones(C_t)) #[C_t]
        tens = np.random.gamma(np.ones((C_t,D)),D*np.ones((C_t,D))) #[C_t,D]
        
        self.set_true_parameter_by_array(mr,tens)
        
        
    def generate_sample(self,N,seed=None):
                
        if N<=0:
            raise TypeError('N must be greater than 0.')
        if seed:
            np.random.seed(seed)
            
        self.N = N
        self.y_t = np.random.multinomial(1,self.mr_t,size=self.N) #[N,C_t]
        self.X_t = np.random.poisson(np.sum(self.tens_t[np.newaxis,:,:]*self.y_t[:,:,np.newaxis],axis=1)) #[N,D]
        
        self.bin_X_t()

        
    def view_data(self,save=False):
        
        fig,axes = plt.subplots(1,self.D,figsize=(6*self.D,6))
        fig.suptitle('data')
        if self.D==1:
            where, degree = np.unique(self.X_t[:,0],return_counts=True)
            axes.bar(where,degree,color=plt.cm.tab10(0))
            axes.set(xlabel=f'x_{0+1}',ylabel='degree')
        if self.D>1:
            for d in range(self.D):
                where, degree = np.unique(self.X_t[:,d],return_counts=True)
                axes[d].bar(where,degree,color=plt.cm.tab10(d))
                axes[d].set(xlabel=f'x_{d+1}',ylabel='degree')
        if save:
            fig.savefig('data_bar_'+re.sub('[ :.-]','',str(datetime.datetime.today()))+'.pdf')
    
        
    def set_model(self,C,cent=None,shape=None,scale=None,seed=None):

        if type(cent)==type(None):
            cent = 10*(np.ones(C)+1/(10**np.arange(C)))
        if type(shape)==type(None):
            shape = np.ones((C,self.D))
        if type(scale)==type(None):
            scale = self.D*np.ones((C,self.D))
        
        if C==np.size(cent) and C==np.size(shape,axis=0) and C==np.size(scale,axis=0)            and self.D==np.size(shape,axis=1) and self.D==np.size(scale,axis=1):
            self.C = C
            self.cent = cent
            self.shape = shape
            self.scale = scale
            self.set_GS(seed=seed)
            self.set_VI()
            self.set_CGS(seed=seed)
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
    
    def GibbsSampling(self,ITER=500,burnin=None,seed=None):
        
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
            
    
    def view_Gibbs_sample(self,save=False):
        
        fig,axes = plt.subplots(self.C,self.D+1,figsize=(6*(self.D+1),6*self.C))
        fig.suptitle('posterior distribution (Gibbs sampling)')
        
        for c in range(self.C):
            degrees,spaces,_ = axes[c,0].hist(self.mr_GS.reshape(-1,self.C)[:,c],bins=round(np.sqrt(np.size(a,axis=0))),linewidth=0,color=plt.cm.tab10(0))
            ytic = np.linspace(0,np.max(degrees),6)
            yticlab = np.linspace(0,np.max(degrees/np.sum((spaces[1]-spaces[0])*degrees)),6)*100//1/100
            axes[c,0].set(xlim=(0,1),xlabel=f'mr_{c}',yticks=ytic,yticklabels=yticlab)
            for d in range(self.D):
                degrees,spaces,_ = axes[c,d+1].hist(self.tens_GS.reshape(-1,self.C,self.D)[:,c,d],bins=round(np.sqrt(np.size(a,axis=0))),linewidth=0,color=plt.cm.tab10(d+1))
                ytic = np.linspace(0,np.max(degrees),6)
                yticlab = np.linspace(0,np.max(degrees/np.sum((spaces[1]-spaces[0])*degrees)),6)*100//1/100
                axes[c,d+1].set(xlim=(0,np.max(self.tens_GS.reshape(-1,self.C,self.D),axis=(0,1))[d]),xlabel=f'tens_{c},{d}',yticks=ytic,yticklabels=yticlab)

        if save:
            fig.savefig('posterior_GS_'+re.sub('[ :.-]','',str(datetime.datetime.today()))+'.pdf')        

  
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
    
    def set_CGS(self,seed=None):
        
        if seed:
            np.random.seed(seed)
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
        
    
    def CollapsedGibbsSampling(self,ITER=500,burnin=None,seed=None):
        
        if seed:
            np.random.seed(seed)
        
        if burnin==None:
            burnin = self.N
        
        print('burn-in...')
        for k in tqdm(range(burnin)):
            self.Collapsed_cycle()
        
        self.y_CGS = np.zeros((ITER,self.N,self.C)) #[I,N,C]
        print('sampling...')
        for k in tqdm(range(ITER)):
            self.Collapsed_cycle()
            self.y_CGS[k,:,:] = self.y_cgsc

    


def try_pmm_model(C_t=2,D=3,N=10000,C=2,ITER_gs=10000,ITER_vi=10000,ITER_cgs=1000,seed=None):


    print('generate model...')
    pm = Poisson_Mixture(C_t,D,N,seed)
    print('done.\ntrue parameter :\n',pm.mr_t,'\n',pm.tens_t)

    print('\nset model...')
    pm.set_model(C,seed)
    print('done.')

    print('\ntry Gibbs sampling...')
    pm.GibbsSampling(ITER_gs)
    print('done.\nGibbs sample mean:\n',np.mean(pm.mr_GS,axis=(0,1)),'\n',np.mean(pm.tens_GS,axis=(0,1)))

    print('\ntry variational inference...')
    pm.VariationalInference(ITER_vi)
    print('done.\nhyper parameter :\n',pm.hp_cent_vic,'\n',pm.hp_shape_vic*pm.hp_scale_vic)
    
    print('\ntry collapsed Gibbs sampling...')
    pm.CollapsedGibbsSampling(ITER_cgs)
    print('done.\nhyper parameter :\n',pm.hp_cent_cgsc,'\n',pm.hp_shape_cgsc/pm.hp_scale_cgsc)
    

