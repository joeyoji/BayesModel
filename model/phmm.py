#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import tools
get_ipython().run_line_magic('matplotlib', 'inline')
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


# In[40]:


class Poisson_Hidden_Markov_generator:
    
    '''
    
    there are two ways to initialize.
    initialization is done optimally for args.
    
    [1] (ipv[np.1darray], tpm[np.2darray], tens[np.2darray]) - three arguments
    
        ipv, tpm and tens mean Initial Probability Vector, Transition Probability Matrix and inTENSity respectively.
        the shape of arguments are S, S x S and S x D respectively, 
        where S is the number of hidden states and D is the dimension of data.
        any element of any arguments must be non-negative.
        the sum of ipv and any row of tpm must be 1 (or less).
        
    [2] (S[int], D[int], (seed[int])) - two (or three) arguments
    
        ipv, tpm and tens are set randomly.
        randomness can be fixed by seed.
    
    '''
    
    def __init__(self,*args):
        
        if len(args)==3:
            if type(args[0])==np.ndarray and type(args[1])==np.ndarray and type(args[2])==np.ndarray:
                self.set_true_directly(args[0],args[1],args[2])
            elif type(args[0])==int and type(args[1])==int and type(args[2])==int:
                self.set_true_randomly(args[0],args[1],args[2])
            else:
                raise TypeError('if length of args is 3, args must be triplets of np.ndarray or int.')
        elif len(args)==2:
            if type(args[0])==int and type(args[1])==int:
                self.set_true_randomly(args[0],args[1])
            else:
                raise TypeError('if length of args is 2, args must be pair of int.')
        else:
            raise TypeError('length of args must be 3 or 2.')
    
    
    def set_true_directly(self,ipv,tpm,tens):
        
        if np.prod(ipv>=0) and np.prod(tpm>=0) and np.prod(tens>=0):
            if np.sum(ipv)<=1 and np.prod(np.sum(tpm,axis=1)<=1):
                if np.size(ipv)==np.size(tpm,axis=0) and np.size(ipv)==np.size(tpm,axis=1) and np.size(ipv)==np.size(tens,axis=0):
                    self.S = np.size(ipv)
                    self.D = np.size(tens,axis=1)
                    self.ipv = ipv
                    self.tpm = tpm
                    self.tens = tens
                else:
                    raise TypeError('the shape of args must be S, S x S and S x D respectively.')
            else:
                raise TypeError('the sum of ipv and any row of tpm must be 1 (or less).')
        else:
            raise TypeError('any element of any arg must be non-negative.')
            
    
    def set_true_randomly(self,S,D,seed=None):
        
        if seed:
            np.random.seed(seed)
        if S>0 and D>0:
            self.S = S
            self.D = D
            self.ipv = np.random.dirichlet(np.ones(self.S))
            self.tpm = np.random.dirichlet(np.ones(self.S),self.S)
            self.tens = np.random.gamma(np.ones((self.S,self.D)),D*np.ones((self.S,self.D)))
        else:
            raise TypeError('S and D must be positive integers.')
    
    
    def generate(self,N,seed=None):
        
        if seed:
            np.random.seed(seed)
            
        if N>0 and type(N)==int:
            self.N = N
            self.y = np.zeros((self.N,self.S),dtype=int)
            self.y[0] = np.random.multinomial(1,self.ipv)
            for n in tqdm(range(1,self.N)):
                self.y[n] = np.random.multinomial(1,np.sum(self.y[n-1,:,np.newaxis]*self.tpm,axis=0))
            self.data = np.random.poisson(np.sum(self.y[:,:,np.newaxis]*self.tens[np.newaxis,:,:],axis=1))
            return self.data
        else:
            raise TypeError('N must be positive integers.')
    
    
    def view_data(self,save=False):
        
        '''
        
        you can visualize data loaded to this model.
        data histogram are marginalized.
        
        ===== arguments =====
        
        [1] save[bool] ... save data graph
        
        '''
        
        fig,axes = plt.subplots(self.D,1,figsize=(0.1*self.N,6*self.D))
        fig.suptitle('data')
        if self.D==1:
            axes.plot(self.data[:,0],marker='d',linewidth=1,color=plt.cm.tab10(0))
            axes.set(ylabel=f'x_{0+1}',xlabel='time')
        else:
            for d in range(self.D):
                axes[d].plot(self.data[:,d],marker='d',linewidth=1,color=plt.cm.tab10(d))
                axes[d].set(ylabel=f'x_{d+1}',xlabel='time')   
        if save:
            fig.savefig('data_plot_'+re.sub('[ :.-]','',str(datetime.datetime.today()))+'.pdf')


# In[53]:


# ipv = np.ones(2)/2
# tpm = np.array([[0.95,0.05],[0.05,0.95]])
# tens = np.array([[6,],[1,]])
# seed = 2022
# N = 200
# phm = Poisson_Hidden_Markov_generator(ipv,tpm,tens)
# phm.generate(N,seed)
# phm.view_data()


# In[3]:


class Hidden_Markov:
    
    
    def set_model(self,C,cent=None,shape=None,scale=None,seed=None):
        
        '''
        
        initialize Gibbs sample, variational approximated distribution, and collapsed Gibbs sample.
        
        ===== arugments =====
        
        [1] C[int] ... the number of model components, which differs from the number of true components(C_t)
        
        [2] cent[np.1darray or None] ... the hyper parameter of Dirichlet prior distribution ( array-shape (C,) )
        
        [3] shape[np.2darray or None] ... one of the hyper parameter of Gamma prior distribution ( array-shape (C,D) )
        
        [4] scale[np.2darray or None] ... the other of the hyper parameter of Gamma prior distribution ( array-shape (C,D) )
        
        [5] seed[int or None] ... random seed to set initial parameter
        
        '''

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
        
        '''
        
        initialize Gibbs sample
        
        ===== arguments =====
        
        [1] L[int] ... the number of parallel sample
        
        [2] seed[int or None] ... random seed to initialize Gibbs sample
        
        '''
              
        if seed:
            np.random.seed(seed)

        self.L = L
        self.hp_pi_gsc = np.ones((self.L,self.N_b,self.C))/self.C #hi_pi_gsc[L,N_b,C]
        
        #run one cycle
        self.Gibbs_cycle()
        
    
    def Gibbs_cycle(self,need_elbo=False):
        
        '''
        
        turn one Gibbs sampling cycle
        
        ===== arguments =====
        
        [1] need_elbo[bool] ... if true, return ELBO(evidence lower bound),else np.nan.
        
        '''
        
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
        self.hp_pi_gsc = (self.mr_gsc[:,np.newaxis,:]*np.prod((                            self.tens_gsc[:,np.newaxis,:,:]**                            self.bin_label[np.newaxis,:,np.newaxis,:])/
                            spsp.factorial(self.bin_label[np.newaxis,:,np.newaxis,:])/\
                            np.exp(self.tens_gsc[:,np.newaxis,:,:]),axis=3)) #[L,N_b,C]
        
        if need_elbo:
            return self.get_elbo('gs')
        else:
            return np.nan
    
    
    def GibbsSampling(self,ITER=500,burnin=None,need_elbo=False,seed=None):
        
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
        self.elbo_GS_trace = np.zeros(ITER) #[I]
        for k in tqdm(range(ITER)):
            self.elbo_GS_trace[k] = self.Gibbs_cycle(need_elbo)
            self.mr_GS[k,:,:] = self.mr_gsc
            self.tens_GS[k,:,:,:] = self.tens_gsc
            
    
    def view_Gibbs_sample(self,save=False):
        
        '''
        
        visiualize marginalized posterior distribution approximated by Gibbs sample
        
        ===== arguments =====
        
        [1] save[bool] ... whether save the graph or not
        
        '''
        
        fig,axes = plt.subplots(self.C,self.D+1,figsize=(6*(self.D+1),6*self.C))
        fig.suptitle('posterior distribution (Gibbs sampling)')
        
        a = self.mr_GS.reshape(-1,self.C)
        B = self.tens_GS.reshape(-1,self.C,self.D)
        
        for c in range(self.C):
            degrees,spaces,_ = axes[c,0].hist(a[:,c],bins=round(np.sqrt(np.size(a,axis=0))),linewidth=0,color=plt.cm.tab10(0))
            ytic = np.linspace(0,np.max(degrees),6)
            yticlab = np.linspace(0,np.max(degrees/np.sum((spaces[1]-spaces[0])*degrees)),6)*100//1/100
            axes[c,0].set(xlim=(0,1),xlabel=f'mr_{c}',yticks=ytic,yticklabels=yticlab)
            for d in range(self.D):
                degrees,spaces,_ = axes[c,d+1].hist(B[:,c,d],bins=round(np.sqrt(np.size(B,axis=0))),linewidth=0,color=plt.cm.tab10(d+1))
                ytic = np.linspace(0,np.max(degrees),6)
                yticlab = np.linspace(0,np.max(degrees/np.sum((spaces[1]-spaces[0])*degrees)),6)*100//1/100
                axes[c,d+1].set(xlim=(0,np.max(B,axis=(0,1))[d]),xlabel=f'tens_{c},{d}',yticks=ytic,yticklabels=yticlab)

        if save:
            fig.savefig('posterior_GS_'+re.sub('[ :.-]','',str(datetime.datetime.today()))+'.pdf')

  
    def set_VI(self):
        
        '''
        
        initialize variational approximated distribution
        
        '''
        
        self.hp_cent_vic = self.cent #[C]
        self.hp_shape_vic = self.shape #[C,D]
        self.hp_scale_vic = self.scale #[C,D]
        
        #run one cycle
        self.Variational_cycle(False)
    
    
    def Variational_cycle(self,need_elbo=True):
        
        '''
        
        run one variational inference cycle
        
        ===== arguments =====
        
        [1] need_elbo[bool] ... if true, reutrn ELBO, or np.nan
        
        '''
        
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
                
        if need_elbo:
            return self.get_elbo('vi')
        else:
            return np.nan
        
    
    def get_elbo(self,method='vi'):
        
        '''
        
        calculate ELBO
        
        ===== arguments =====
        
        [1] method ['gs', 'vi', or 'cgs'] ... designate method
        
        '''
        
        if method=='gs':
            hpi = self.hp_pi_gsc[0]/np.sum(self.hp_pi_gsc[0],axis=1)[:,np.newaxis] #[N_b,C]
            hcent = self.hp_cent_gsc[0] #[C]
            hshape = self.hp_shape_gsc[0] #[C,D]
            hscale = self.hp_scale_gsc[0] #[C,D]
            bwei = self.bin_weight #[N_b,C]
            blab = self.bin_label #[N_b,D]
        elif method=='vi':
            hpi = self.hp_pi_vic
            hcent = self.hp_cent_vic
            hshape = self.hp_shape_vic
            hscale = self.hp_scale_vic
            bwei = self.bin_weight
            blab = self.bin_label
        elif method=='cgs':
            hpi = self.hp_pi_cgsc/np.sum(self.hp_pi_cgsc,axis=1)[:,np.newaxis] #[N,C]
            hcent = self.hp_cent_cgsc
            hshape = self.hp_shape_cgsc
            hscale = self.hp_scale_cgsc
            bwei = 1 #[N,C]
            blab = self.X_t #[N,D]
        else:
            raise ValueError("you can designate only 'gs', 'vi', and 'cgs' as method.")
            
        lmd_ex = hshape*hscale #[C,D]
        ln_lmd_ex = spsp.digamma(hshape)+np.log(hscale) #[C,D]
        ln_pi_ex = spsp.digamma(hcent)-spsp.digamma(np.sum(hcent)) #[C]
        
        #calculate poisson mixture elbo
        ln_p_poi = np.sum(bwei*np.sum(hpi*np.sum(blab[:,np.newaxis,:]*ln_lmd_ex[np.newaxis,:,:]-                      np.log(spsp.factorial(blab[:,np.newaxis,:]))-lmd_ex[np.newaxis,:,:],axis=2),axis=1))
        ln_p_cat = np.sum(bwei*np.sum(hpi*ln_pi_ex[np.newaxis,:],axis=1))
        ln_q_cat = np.sum(bwei*np.sum(hpi*np.log(hpi),axis=1))
        kl_qp_pi = np.sum(spsp.loggamma(self.cent))-spsp.loggamma(np.sum(self.cent))-                    (np.sum(spsp.loggamma(hcent))-spsp.loggamma(np.sum(hcent)))+                    np.sum((hcent-self.cent)*ln_pi_ex)
        kl_qp_lambda = np.sum(spsp.loggamma(self.shape)+self.shape*np.log(self.scale))-                        np.sum(spsp.loggamma(hshape)+hshape*np.log(hscale))+                        np.sum((hshape-self.shape)*ln_lmd_ex)+                        np.sum((1/self.scale-1/hscale)*lmd_ex)

        ELBO = ln_p_poi+ln_p_cat-ln_q_cat-kl_qp_lambda-kl_qp_pi

        return ELBO
    
    
    def view_elbo(self,save=False):
        
        '''
        
        visualize ELBO transition
        
        ===== arguments =====
        
        [1] save[bool] ... whether save the graph or not
        
        '''
        
        fig,ax = plt.subplots(1,1,figsize=(6,6))
        ax.plot(self.elbo_GS_trace,label='GS')
        ax.plot(self.elbo_VI_trace,label='VI')
        ax.plot(self.elbo_CGS_trace,label='CGS')
        ax.legend()
        ax.set(title='ELBO',xlabel='iteration',xscale='log')
        
    
    
    def VariationalInference(self,ITER=500,need_elbo=True):
        
        '''
        
        run variational inference
        
        ===== arguments =====
        
        [1] ITER[int] ... the number of times to run variational infernce
        
        [2] need_elbo[bool] ... whether return ELBO or not
        
        '''
        
        self.hp_cent_VI_trace = np.zeros((ITER,self.C)) #[I,C]
        self.hp_shape_VI_trace = np.zeros((ITER,self.C,self.D)) #[I,C,D]
        self.hp_scale_VI_trace = np.zeros((ITER,self.C,self.D)) #[I,C,D]
        self.elbo_VI_trace = np.zeros(ITER) #[I]
        
        for k in tqdm(range(ITER)):
            self.elbo_VI_trace[k] = self.Variational_cycle(need_elbo)
            self.hp_cent_VI_trace[k,:] = self.hp_cent_vic
            self.hp_shape_VI_trace[k,:,:] = self.hp_shape_vic
            self.hp_scale_VI_trace[k,:,:] = self.hp_scale_vic
    
    
    def view_Variational_trace(self,save=False):
        
        '''
        
        visualize the transition trace of variational approximated distribution
        
        ===== arguments =====
        
        [1] save[bool] ... whether save the graph or not
        
        '''
        
        fig,axes = plt.subplots(self.C,self.D+1,figsize=(6*(self.D+1),6*self.C))
        fig.suptitle('parameter transition of variational inference')

        a = self.hp_cent_VI_trace
        B = self.hp_shape_VI_trace*self.hp_scale_VI_trace

        for c in range(self.C):
            axes[c,0].plot(a[:,c],color=plt.cm.tab10(0))
            axes[c,0].set(xlabel='iteration',ylabel=f'mr_{c}',ylim=(np.min(a)*0.9,np.max(a)*1.05))
            for d in range(self.D):
                axes[c,d+1].plot(B[:,c,d],color=plt.cm.tab10(d+1))
                axes[c,d+1].set(xlabel='iteration',ylabel=f'tens_{c},{d}',ylim=(np.min(B,axis=(0,1))[d]*0.9,np.max(B,axis=(0,1))[d]*1.05))

        if save:
            fig.savefig('transition_VI_'+re.sub('[ :.-]','',str(datetime.datetime.today()))+'.pdf')
    
    
    def set_CGS(self,seed=None):
        
        '''
        
        initialize the collapsed Gibbs sample
        
        ===== arguments =====
        
        [1] seed[int or None] ... random seed to initialize collapsed Gibbs sample
        
        '''
        
        if seed:
            np.random.seed(seed)
        #before>>>
        self.y_cgsc = np.random.multinomial(1,np.ones(self.C)/self.C,size=self.N) #[N,C]
        self.hp_cent_cgsc = np.sum(self.y_cgsc,axis=0)+self.cent #[C]
        self.hp_shape_cgsc = np.sum(self.y_cgsc[:,:,np.newaxis]*self.X_t[:,np.newaxis,:],axis=0)+self.shape #[C,D]
        self.hp_scale_cgsc = 1/(1/self.scale+np.sum(self.y_cgsc,axis=0)[:,np.newaxis]) #[C,D]
        #<<<
        
        self.Collapsed_cycle()
    
    
    def Collapsed_cycle(self,need_elbo=False):
        
        '''
        
        run one collapsed Gibbs sampling cycle
        
        ===== arguments =====
        
        [1] need_elbo[bool] ... if true, return ELBO, or np.nan
        
        '''
                
        self.hp_cent_cgsc = self.hp_cent_cgsc[np.newaxis,:]-self.y_cgsc #[N,C]
        self.hp_shape_cgsc = self.hp_shape_cgsc[np.newaxis,:,:]-self.y_cgsc[:,:,np.newaxis]*self.X_t[:,np.newaxis,:] #[N,C,D]
        self.hp_scale_cgsc = 1/(1/self.hp_scale_cgsc[np.newaxis,:,:]-self.y_cgsc[:,:,np.newaxis]) #[N,C,D]

        nb_p = 1/(1+1/self.hp_scale_cgsc) #[N,C,D]
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
        self.hp_scale_cgsc = 1/(1/self.scale+np.sum(self.y_cgsc,axis=0)[:,np.newaxis]) #[C,D]
        
        if need_elbo:
            return self.get_elbo('cgs')
        else:
            return np.nan
        
    
    def CollapsedGibbsSampling(self,ITER=500,burnin=None,need_elbo=False,seed=None):
        
        '''
        
        run collapsed Gibbs sampling
        
        ===== arguments =====
        
        [1] ITER[int] ... the number of times to run CGS for sampling
        
        [2] burnin[int] ... the number of times to run CGS for stabilizing sample
        
        [3] need_elbo[bool] ... whether return ELBO or not
        
        [4] seed[int or None] ... random seed to run collapsed Gibbs sampling
        
        
        '''
        
        if seed:
            np.random.seed(seed)
        
        if burnin==None:
            burnin = self.N
        
        print('burn-in...')
        for k in tqdm(range(burnin)):
            self.Collapsed_cycle()
        
        self.y_CGS = np.zeros((ITER,self.N,self.C)) #[I,N,C]
        self.elbo_CGS_trace = np.zeros(ITER) #[I]
        print('sampling...')
        for k in tqdm(range(ITER)):
            self.elbo_CGS_trace[k] = self.Collapsed_cycle(need_elbo)
            self.y_CGS[k,:,:] = self.y_cgsc

    
    def view_Collapsed_assignment(self,save=False):
        
        '''
        
        visualize CGS estimated label assignment on data
        
        ===== arguments =====
        
        [1] save[bool] ... whether save the graph or not
        
        '''
        
        fig,axes = plt.subplots(1,self.D,figsize=(6*self.D,6))
        fig.suptitle('component assignment')
        for d in range(self.D):
            x = self.X_t[:,d]
            y = self.y_cgsc
            xlab = np.unique(x)
            ydegree = np.add.accumulate(np.sum((y[:,np.newaxis,:]*(x[:,np.newaxis]==xlab)[:,:,np.newaxis]),axis=0),axis=1)
            for c in range(self.C):
                axes[d].bar(xlab,ydegree[:,self.C-c-1],linewidth=0,label=f'component {c}')
                axes[d].set(ylabel='degree',xlabel=f'x_{d}')
                if d==0:
                    axes[d].legend()
        if save:
            fig.savefig('transition_VI_'+re.sub('[ :.-]','',str(datetime.datetime.today()))+'.pdf')
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




