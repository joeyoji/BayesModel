#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import tools
# %matplotlib inline
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


# In[2]:


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
        
        fig,axes = plt.subplots(self.D,1,figsize=(np.min([0.1*self.N,100]),6*self.D))
        fig.suptitle('data')
        if self.D==1:
            axes.plot(self.data[:,0],marker='d',linewidth=1,color=plt.cm.tab10(0))
            axes.set(ylabel=f'x_{0+1}',xlabel='time')
        else:
            for d in range(self.D):
                axes[d].plot(self.data[:,d],marker='d',linewidth=1,color=plt.cm.tab10(d))
                axes[d].set(ylabel=f'x_{d+1}',xlabel='time')   
        if save:
            fig.savefig('data_plot_'+re.sub('[ :.-]','',str(datetime.datetime.today()))+'.pdf',bbox_inches='tight', pad_inches=0)


# In[3]:


class Poisson_Hidden_Markov:
    
    
    def __init__(self,data,S,cent_ipv=None,cent_tpm=None,shape=None,scale=None,seed=None):
        
        self.load_data(data)
        self.set_model(S,cent_ipv,cent_tpm,shape,scale,seed)
        
    
    
    def load_data(self,data):
        
        if type(data)==np.ndarray:
            if len(data.shape)==2:
                self.data = data
                self.N = np.size(data,axis=0)
                self.D = np.size(data,axis=1)
            else:
                raise TypeError('data must be N x D, where N is the length of data series and D is the dimension of data.')
        else:
            raise TypeError('type of data should be np.2darray.')
        
        
    def set_model(self,S,cent_ipv=None,cent_tpm=None,shape=None,scale=None,seed=None):
        
        '''
        
        initialize completely factorized and structured variational approximated distribution.
        
        ===== arugments =====
        
        [1] S[int] ... the number of hidden states of the model.
        
        [2] cent_ipv[np.1darray] ... the hyper parameter of Dirichlet prior distribution (initial probability vector).
        
        [3] cent_tpm[np.2darray] ... the hyper parameter of Dirichlet prior distribution (transition probability matrix).
        
        [3] shape[np.2darray] ... one of the hyper parameter of Gamma prior distribution ( array-shape (S,D) )
        
        [4] scale[np.2darray] ... the other of the hyper parameter of Gamma prior distribution ( array-shape (S,D) )
        
        [5] seed[int] ... random seed to set initial parameter
        
        '''
        if seed:
            np.random.seed(seed)
            
        if type(cent_ipv)==type(None):
            cent_ipv = np.random.uniform(size=S)
        if type(cent_tpm)==type(None):
            cent_tpm = np.random.uniform(size=(S,S))
        if type(shape)==type(None):
            shape = 2*np.random.uniform(size=(S,self.D))
        if type(scale)==type(None):
            scale = 2*self.D*np.random.uniform(size=(S,self.D))
        
        if S==np.size(cent_ipv) and S==np.size(cent_tpm,axis=0) and S==np.size(cent_tpm,axis=1)                and S==np.size(shape,axis=0) and S==np.size(scale,axis=0):
            if np.size(shape,axis=1)==np.size(scale,axis=1):
                self.S = S
                self.cent_ipv = cent_ipv
                self.cent_tpm = cent_tpm
                self.shape = shape
                self.scale = scale
                self.set_VI()
            else:
                raise TypeError('the dimension of shape and scale are not corresponding.')
        else:
            raise TypeError('the number of hidden states are not corresponding among parameters.')


    def set_VI(self):
        
        self.set_cfVI()
        self.set_sVI()
            

    def set_cfVI(self):
        
        '''
        
        initialize (completely factorized) variational approximated distribution.
        
        '''
        
        self.cent_ipv_cfvi = self.cent_ipv #[S]
        self.cent_tpm_cfvi = self.cent_tpm #[S,S]
        self.shape_cfvi = self.shape #[S,D]
        self.scale_cfvi = self.scale #[S,D]
        self.eta_cfvi = np.random.dirichlet(np.ones(self.S),size=self.N) #[N,S]
        
        #run one cycle
        self.cfVariational_cycle(False)
    
    
    def cfVariational_cycle(self,need_elbo=False):
        
        '''
        
        run one variational inference cycle
        
        ===== arguments =====
        
        [1] need_elbo[bool] ... if true, reutrn ELBO, or np.nan　(currently not working)
        
        '''
        
        #calculate expected params
        ex_s = self.eta_cfvi #[N,S]
        ex_lmd = self.shape_cfvi*self.scale_cfvi #[S,D]
        ex_ln_lmd = spsp.digamma(self.shape_cfvi)+np.log(self.scale_cfvi) #[S,D]
        ex_ln_pi = spsp.digamma(self.cent_ipv_cfvi)-spsp.digamma(np.sum(self.cent_ipv_cfvi)) #[S]
        ex_ln_A = spsp.digamma(self.cent_tpm_cfvi)-spsp.digamma(np.sum(self.cent_tpm_cfvi,axis=1)) #[S,S]
        
        #cauclate hyper params
        #hyper params of Gamma
        self.shape_cfvi = np.sum(ex_s[:,:,np.newaxis]*self.data[:,np.newaxis,:],axis=0)+self.shape #[S,D]
        self.scale_cfvi = 1/(np.sum(ex_s,axis=0)[:,np.newaxis]+1/self.scale) #[S,D]
        #hyper params of Dirichlet ( Initial Probability Vector )
        self.cent_ipv_cfvi = ex_s[0]+self.cent_ipv #[S]
        #hyper params of Dirichlet ( Transition Probability Matrix )
        self.cent_tpm_cfvi = np.sum(ex_s[:-1,:,np.newaxis]*ex_s[1:,np.newaxis,:],axis=0)+self.cent_tpm #[S,S]
        #hyper params of Categorical ( Initial States )
        self.eta_cfvi[0,:] = np.exp(np.sum(self.data[0,np.newaxis,:]*ex_ln_lmd-ex_lmd,axis=1)+                                    ex_ln_pi+np.sum(ex_ln_A*self.eta_cfvi[1,np.newaxis,:],axis=1)) #[S]
        self.eta_cfvi[0,:] /= np.sum(self.eta_cfvi[0,:])
        #hyper params of Categorical ( Midterm States )
        self.eta_cfvi[1:-1,:] = np.exp(np.sum(self.data[1:-1,np.newaxis,:]*ex_ln_lmd-ex_lmd,axis=2)+                                    np.sum(self.eta_cfvi[:-2,:,np.newaxis]*ex_ln_A[np.newaxis,:,:],axis=1)+                                    np.sum(ex_ln_A[np.newaxis,:,:]*self.eta_cfvi[2:,np.newaxis,:],axis=2)) #[N-2,S]
        self.eta_cfvi[1:-1,:] /= np.sum(self.eta_cfvi[1:-1,:],axis=1)[:,np.newaxis]
        #hyper params of Categorical ( Final States )
        self.eta_cfvi[-1,:] = np.exp(np.sum(self.data[-1,np.newaxis,:]*ex_ln_lmd-ex_lmd,axis=1)+                                    np.sum(self.eta_cfvi[-1,:,np.newaxis]*ex_ln_A,axis=0)) #[S]
        self.eta_cfvi[-1,:] /= np.sum(self.eta_cfvi[-1,:])
        
        # if need_elbo:
        #     return self.get_elbo('vi')
        # else:
        #     return np.nan

        
    def cfVI(self,ITER=500,need_elbo=False):
        
        '''
        
        run variational inference
        
        ===== arguments =====
        
        [1] ITER[int] ... the number of times to run variational infernce
        
        [2] need_elbo[bool] ... whether return ELBO or not ( currently not working )
        
        '''
        
        self.cent_ipv_cfVI_trace = np.zeros((ITER,self.S)) #[I,S]
        self.cent_tpm_cfVI_trace = np.zeros((ITER,self.S,self.S)) #[I,S,S]
        self.shape_cfVI_trace = np.zeros((ITER,self.S,self.D)) #[I,S,D]
        self.scale_cfVI_trace = np.zeros((ITER,self.S,self.D)) #[I,S,D]
        self.elbo_cfVI_trace = np.zeros(ITER) #[I]
        
        for k in tqdm(range(ITER)):
            self.cfVariational_cycle(need_elbo)
            self.cent_ipv_cfVI_trace[k,:] = self.cent_ipv_cfvi
            self.cent_tpm_cfVI_trace[k,:,:] = self.cent_tpm_cfvi
            self.shape_cfVI_trace[k,:,:] = self.shape_cfvi
            self.scale_cfVI_trace[k,:,:] = self.scale_cfvi
    
    
    def view_cfVI_trace(self,save=False):
        
        '''
        
        visualize the transition trace of variational approximated distribution
        
        ===== arguments =====
        
        [1] save[bool] ... whether save the graph or not
        
        '''
        
        fig,axes = plt.subplots(self.S,1+self.S+self.D,figsize=(6*(1+self.S+self.D),6*self.S))
        fig.suptitle('parameter transition of variational inference')

        a = self.cent_ipv_cfVI_trace
        A = self.cent_tpm_cfVI_trace
        B = self.shape_cfVI_trace*self.scale_cfVI_trace

        for c in range(self.S):
            axes[c,0].plot(a[:,c],color=plt.cm.tab10(0))
            axes[c,0].set(xlabel='iteration',ylabel=f'alpha_{c}',ylim=(np.min(a)*0.9,np.max(a)*1.05))
            for s in range(self.S):
                axes[c,s+1].plot(A[:,c,s],color=plt.cm.tab10(s+1))
                axes[c,s+1].set(xlabel='iteration',ylabel=f'beta_{c},{s}',                                ylim=(np.min(A,axis=(0,1))[s]*0.9,np.max(A,axis=(0,1))[s]*1.05))
            for d in range(self.D):
                axes[c,1+self.S+d].plot(B[:,c,d],color=plt.cm.tab10(1+self.S+d))
                axes[c,1+self.S+d].set(xlabel='iteration',ylabel=f'tens_{c},{d}',                                       ylim=(np.min(B,axis=(0,1))[d]*0.9,np.max(B,axis=(0,1))[d]*1.05))

        if save:
            fig.savefig('transition_VI_'+re.sub('[ :.-]','',str(datetime.datetime.today()))+'.pdf')
            

    def set_sVI(self):
        
        '''
        
        initialize (structured) variational approximated distribution.
        
        '''
        
        self.cent_ipv_svi = self.cent_ipv #[S]
        self.cent_tpm_svi = self.cent_tpm #[S,S]
        self.shape_svi = self.shape #[S,D]
        self.scale_svi = self.scale #[S,D]
        self.eta_svi = np.random.dirichlet(np.ones(self.S),size=self.N) #[N,S]
        self.etaeta_svi = np.random.dirichlet(np.ones(self.S),size=(self.N-1,self.S)) #[N-1,S,S]
        
        #run one cycle
        self.sVariational_cycle()   
        
        
    def sVariational_cycle(self):
        
        #calculate expected params
        ex_lmd = self.shape_svi*self.scale_svi #[S,D]
        ex_ln_lmd = spsp.digamma(self.shape_svi)+np.log(self.scale_svi) #[S,D]
        ex_ln_pi = spsp.digamma(self.cent_ipv_svi)-spsp.digamma(np.sum(self.cent_ipv_svi)) #[S]
        ex_ln_A = spsp.digamma(self.cent_tpm_svi)-spsp.digamma(np.sum(self.cent_tpm_svi,axis=1)) #[S,S]
        #calculate expected hidden variables
        ex_pss = np.exp(ex_ln_A) #[S,S]
        ex_pxs = np.exp(np.sum(self.data[:,np.newaxis,:]*ex_ln_lmd[np.newaxis,:,:]-ex_lmd[np.newaxis,:,:],axis=2)) #[N,S]
        forward = np.zeros((self.N,self.S)) #[N,S]
        backward = np.zeros((self.N,self.S)) #[N,S]
        forward[0,:] = ex_pxs[0]*np.exp(ex_ln_pi) #[S]
        forward[0,:] /= np.sum(forward[0,:])
        backward[-1,:] = np.ones(self.S)/self.S #[S]
        for n in range(1,self.N):
            forward[n,:] = ex_pxs[n,:]*np.sum(ex_pss*forward[n-1,:,np.newaxis],axis=0) #[S]
            forward[n,:] /= np.sum(forward[n,:])
            backward[-n-1,:] = np.sum(ex_pxs[-n,np.newaxis,:]*ex_pss*backward[-n,np.newaxis,:],axis=1) #[S]
            backward[-n-1,:] /= np.sum(backward[-n-1,:])
        #calculated expected hidden variables
        ex_s = forward*backward
        ex_s /= np.sum(ex_s,axis=1)[:,np.newaxis] #[N,S]
        ex_ss = ex_pxs[1:,np.newaxis,:]*ex_pss[np.newaxis,:,:]*forward[:-1,:,np.newaxis]*backward[1:,np.newaxis,:] #[N-1,S,S]
        ex_ss /= np.sum(ex_ss,axis=(1,2))[:,np.newaxis,np.newaxis]
        
        #renew cauclate hyper params
        #hyper params of Gamma
        self.shape_svi = np.sum(ex_s[:,:,np.newaxis]*self.data[:,np.newaxis,:],axis=0)+self.shape #[S,D]
        self.scale_svi = 1/(np.sum(ex_s,axis=0)[:,np.newaxis]+1/self.scale) #[S,D]
        #hyper params of Dirichlet ( Initial Probability Vector )
        self.cent_ipv_svi = ex_s[0,:]+self.cent_ipv #[S]
        #hyper params of Dirichlet ( Transition Probability Matrix )
        self.cent_tpm_svi = np.sum(ex_ss,axis=0)+self.cent_tpm #[S,S]
        #hyper params of Categorical
        self.eta_svi = ex_s #[N,S]
        self.etaeta_svi = ex_ss #[N-1,S,S]
            

    def sVI(self,ITER=500):
        
        '''
        
        run variational inference
        
        ===== arguments =====
        
        [1] ITER[int] ... the number of times to run variational infernce
        
        [2] need_elbo[bool] ... whether return ELBO or not ( currently not working )
        
        '''
        
        self.cent_ipv_sVI_trace = np.zeros((ITER,self.S)) #[I,S]
        self.cent_tpm_sVI_trace = np.zeros((ITER,self.S,self.S)) #[I,S,S]
        self.shape_sVI_trace = np.zeros((ITER,self.S,self.D)) #[I,S,D]
        self.scale_sVI_trace = np.zeros((ITER,self.S,self.D)) #[I,S,D]
        self.elbo_sVI_trace = np.zeros(ITER) #[I]
        
        for k in tqdm(range(ITER)):
            self.sVariational_cycle()
            self.cent_ipv_sVI_trace[k,:] = self.cent_ipv_svi
            self.cent_tpm_sVI_trace[k,:,:] = self.cent_tpm_svi
            self.shape_sVI_trace[k,:,:] = self.shape_svi
            self.scale_sVI_trace[k,:,:] = self.scale_svi            
            
            
    def view_sVI_trace(self,save=False):
        
        '''
        
        visualize the transition trace of variational approximated distribution
        
        ===== arguments =====
        
        [1] save[bool] ... whether save the graph or not
        
        '''
        
        fig,axes = plt.subplots(self.S,1+self.S+self.D,figsize=(6*(1+self.S+self.D),6*self.S))
        fig.suptitle('parameter transition of variational inference')

        a = self.cent_ipv_sVI_trace
        A = self.cent_tpm_sVI_trace
        B = self.shape_sVI_trace*self.scale_sVI_trace

        for c in range(self.S):
            axes[c,0].plot(a[:,c],color=plt.cm.tab10(0))
            axes[c,0].set(xlabel='iteration',ylabel=f'alpha_{c}',ylim=(np.min(a)*0.9,np.max(a)*1.05))
            for s in range(self.S):
                axes[c,s+1].plot(A[:,c,s],color=plt.cm.tab10(s+1))
                axes[c,s+1].set(xlabel='iteration',ylabel=f'beta_{c},{s}',                                ylim=(np.min(A,axis=(0,1))[s]*0.9,np.max(A,axis=(0,1))[s]*1.05))
            for d in range(self.D):
                axes[c,1+self.S+d].plot(B[:,c,d],color=plt.cm.tab10(1+self.S+d))
                axes[c,1+self.S+d].set(xlabel='iteration',ylabel=f'tens_{c},{d}',                                       ylim=(np.min(B,axis=(0,1))[d]*0.9,np.max(B,axis=(0,1))[d]*1.05))

        if save:
            fig.savefig('transition_VI_'+re.sub('[ :.-]','',str(datetime.datetime.today()))+'.pdf')
            
            
            
            
            
            

    def view_elbo(self,save=False):
        
        pass
#         '''
        
#         visualize ELBO transition
        
#         ===== arguments =====
        
#         [1] save[bool] ... whether save the graph or not
        
#         '''
        
#         fig,ax = plt.subplots(1,1,figsize=(6,6))
#         ax.plot(self.elbo_GS_trace,label='GS')
#         ax.plot(self.elbo_VI_trace,label='VI')
#         ax.plot(self.elbo_CGS_trace,label='CGS')
#         ax.legend()
#         ax.set(title='ELBO',xlabel='iteration',xscale='log')

    def get_elbo(self,method='vi'):
        
        pass
#         '''
        
#         calculate ELBO
        
#         ===== arguments =====
        
#         [1] method ['gs', 'vi', or 'cgs'] ... designate method
        
#         '''
        
#         if method=='gs':
#             hpi = self.hp_pi_gsc[0]/np.sum(self.hp_pi_gsc[0],axis=1)[:,np.newaxis] #[N_b,C]
#             hcent = self.hp_cent_gsc[0] #[C]
#             hshape = self.hp_shape_gsc[0] #[C,D]
#             hscale = self.hp_scale_gsc[0] #[C,D]
#             bwei = self.bin_weight #[N_b,C]
#             blab = self.bin_label #[N_b,D]
#         elif method=='vi':
#             hpi = self.hp_pi_vic
#             hcent = self.hp_cent_vic
#             hshape = self.hp_shape_vic
#             hscale = self.hp_scale_vic
#             bwei = self.bin_weight
#             blab = self.bin_label
#         elif method=='cgs':
#             hpi = self.hp_pi_cgsc/np.sum(self.hp_pi_cgsc,axis=1)[:,np.newaxis] #[N,C]
#             hcent = self.hp_cent_cgsc
#             hshape = self.hp_shape_cgsc
#             hscale = self.hp_scale_cgsc
            # bwei = 1 #[N,C]
#             blab = self.X_t #[N,D]
#         else:
#             raise ValueError("you can designate only 'gs', 'vi', and 'cgs' as method.")
            
#         lmd_ex = hshape*hscale #[C,D]
#         ln_lmd_ex = spsp.digamma(hshape)+np.log(hscale) #[C,D]
#         ln_pi_ex = spsp.digamma(hcent)-spsp.digamma(np.sum(hcent)) #[C]
        
#         #calculate poisson mixture elbo
#         ln_p_poi = np.sum(bwei*np.sum(hpi*np.sum(blab[:,np.newaxis,:]*ln_lmd_ex[np.newaxis,:,:]-\
#                       np.log(spsp.factorial(blab[:,np.newaxis,:]))-lmd_ex[np.newaxis,:,:],axis=2),axis=1))
#         ln_p_cat = np.sum(bwei*np.sum(hpi*ln_pi_ex[np.newaxis,:],axis=1))
#         ln_q_cat = np.sum(bwei*np.sum(hpi*np.log(hpi),axis=1))
#         kl_qp_pi = np.sum(spsp.loggamma(self.cent))-spsp.loggamma(np.sum(self.cent))-\
#                     (np.sum(spsp.loggamma(hcent))-spsp.loggamma(np.sum(hcent)))+\
#                     np.sum((hcent-self.cent)*ln_pi_ex)
#         kl_qp_lambda = np.sum(spsp.loggamma(self.shape)+self.shape*np.log(self.scale))-\
#                         np.sum(spsp.loggamma(hshape)+hshape*np.log(hscale))+\
#                         np.sum((hshape-self.shape)*ln_lmd_ex)+\
#                         np.sum((1/self.scale-1/hscale)*lmd_ex)

#         ELBO = ln_p_poi+ln_p_cat-ln_q_cat-kl_qp_lambda-kl_qp_pi

#         return ELBO
    

    
    
    
    
    
    def set_GS(self,L=1,seed=None):
        
        pass
#         '''
        
#         initialize Gibbs sample
        
#         ===== arguments =====
        
#         [1] L[int] ... the number of parallel sample
        
#         [2] seed[int or None] ... random seed to initialize Gibbs sample
        
#         '''
              
#         if seed:
#             np.random.seed(seed)

#         self.L = L
#         self.hp_pi_gsc = np.ones((self.L,self.N_b,self.C))/self.C #hi_pi_gsc[L,N_b,C]
        
#         #run one cycle
#         self.Gibbs_cycle()
        
    
    def Gibbs_cycle(self,need_elbo=False):
        
        pass
#         '''
        
#         turn one Gibbs sampling cycle
        
#         ===== arguments =====
        
#         [1] need_elbo[bool] ... if true, return ELBO(evidence lower bound),else np.nan.
        
#         '''
        
#         self.y_gsc = np.zeros((self.L,self.N_b,self.C),dtype=int) #[L,N_b,C]
#         Wei = np.repeat([np.copy(self.bin_weight),],self.L,axis=0) #Wei[L,N_b]
#         for c in range(self.C-1):
#             prb = self.hp_pi_gsc[:,:,c]/np.sum(self.hp_pi_gsc[:,:,c:],axis=2)
#             prb[np.isnan(prb)] = 0
#             self.y_gsc[:,:,c] = np.random.binomial(Wei,prb)
#             Wei-=self.y_gsc[:,:,c]
#         self.y_gsc[:,:,self.C-1] = Wei
        
#         #set hp_cent_gsc & mr_gsc
#         self.mr_gsc = np.zeros((self.L,self.C)) #[L,C]
#         self.hp_cent_gsc = np.sum(self.y_gsc,axis=1)+self.cent #[L,C]
#         remains = np.ones(self.L)
#         for c in range(self.C-1):
#             self.mr_gsc[:,c] = remains*np.random.beta(self.hp_cent_gsc[:,c],\
#                                                       np.sum(self.hp_cent_gsc[:,c+1:],axis=1))
#             remains-=self.mr_gsc[:,c]
#         self.mr_gsc[:,self.C-1] = remains
        
#         #set hp_shape_gsc,hp_scale_gsc and tens_gsc
#         self.hp_shape_gsc = np.sum(self.y_gsc[:,:,:,np.newaxis]*\
#                               self.bin_label[np.newaxis,:,np.newaxis,:]\
#                               ,axis=1)+self.shape[np.newaxis,:,:] #[L,C,D]
#         self.hp_scale_gsc = self.scale[np.newaxis,:,:]/\
#                             (1+np.sum(self.y_gsc,axis=1)[:,:,np.newaxis]*\
#                             self.scale[np.newaxis,:,:]) #[L,C,D]
#         self.tens_gsc = np.random.gamma(self.hp_shape_gsc,self.hp_scale_gsc) #[L,C,D]
        
#         #calculate hp_pi_gsc
#         self.hp_pi_gsc = (self.mr_gsc[:,np.newaxis,:]*np.prod((\
#                             self.tens_gsc[:,np.newaxis,:,:]**\
#                             self.bin_label[np.newaxis,:,np.newaxis,:])/
#                             spsp.factorial(self.bin_label[np.newaxis,:,np.newaxis,:])/\
#                             np.exp(self.tens_gsc[:,np.newaxis,:,:]),axis=3)) #[L,N_b,C]
        
#         if need_elbo:
#             return self.get_elbo('gs')
#         else:
#             return np.nan
    
    
    def GibbsSampling(self,ITER=500,burnin=None,need_elbo=False,seed=None):
        
        pass
#         '''
        
#         run Gibbs sampling
        
#         ===== arguments =====
        
#         ITER[int] ... the number of times to turn GS for sampling
        
#         burnin[int or None] ... the number of times to turn GS for stabilizing sample
        
#         need_elbo[bool] ... whether return ELBO or not
        
#         seed[int or None] ... random seed for Gibbs sampling
        
#         '''
        
#         if seed:
#             np.random.seed(seed)
            
#         if burnin==None:
#             burnin = self.N
            
#         print('burn-in...')
#         for k in tqdm(range(burnin)):
#             self.Gibbs_cycle()
        
#         print('sampling...')
#         self.mr_GS = np.zeros((ITER,self.L,self.C)) #[I,L,C]
#         self.tens_GS = np.zeros((ITER,self.L,self.C,self.D)) #[I,L,C,D]
#         self.elbo_GS_trace = np.zeros(ITER) #[I]
#         for k in tqdm(range(ITER)):
#             self.elbo_GS_trace[k] = self.Gibbs_cycle(need_elbo)
#             self.mr_GS[k,:,:] = self.mr_gsc
#             self.tens_GS[k,:,:,:] = self.tens_gsc
            
    
    def view_Gibbs_sample(self,save=False):
        
        pass
#         '''
        
#         visiualize marginalized posterior distribution approximated by Gibbs sample
        
#         ===== arguments =====
        
#         [1] save[bool] ... whether save the graph or not
        
#         '''
        
#         fig,axes = plt.subplots(self.C,self.D+1,figsize=(6*(self.D+1),6*self.C))
#         fig.suptitle('posterior distribution (Gibbs sampling)')
        
#         a = self.mr_GS.reshape(-1,self.C)
#         B = self.tens_GS.reshape(-1,self.C,self.D)
        
#         for c in range(self.C):
#             degrees,spaces,_ = axes[c,0].hist(a[:,c],bins=round(np.sqrt(np.size(a,axis=0))),linewidth=0,color=plt.cm.tab10(0))
#             ytic = np.linspace(0,np.max(degrees),6)
#             yticlab = np.linspace(0,np.max(degrees/np.sum((spaces[1]-spaces[0])*degrees)),6)*100//1/100
#             axes[c,0].set(xlim=(0,1),xlabel=f'mr_{c}',yticks=ytic,yticklabels=yticlab)
#             for d in range(self.D):
#                 degrees,spaces,_ = axes[c,d+1].hist(B[:,c,d],bins=round(np.sqrt(np.size(B,axis=0))),linewidth=0,color=plt.cm.tab10(d+1))
#                 ytic = np.linspace(0,np.max(degrees),6)
#                 yticlab = np.linspace(0,np.max(degrees/np.sum((spaces[1]-spaces[0])*degrees)),6)*100//1/100
#                 axes[c,d+1].set(xlim=(0,np.max(B,axis=(0,1))[d]),xlabel=f'tens_{c},{d}',yticks=ytic,yticklabels=yticlab)

#         if save:
#             fig.savefig('posterior_GS_'+re.sub('[ :.-]','',str(datetime.datetime.today()))+'.pdf')

  

    
    
    def set_CGS(self,seed=None):
        
        pass
#         '''
        
#         initialize the collapsed Gibbs sample
        
#         ===== arguments =====
        
#         [1] seed[int or None] ... random seed to initialize collapsed Gibbs sample
        
#         '''
        
#         if seed:
#             np.random.seed(seed)
#         #before>>>
#         self.y_cgsc = np.random.multinomial(1,np.ones(self.C)/self.C,size=self.N) #[N,C]
#         self.hp_cent_cgsc = np.sum(self.y_cgsc,axis=0)+self.cent #[C]
#         self.hp_shape_cgsc = np.sum(self.y_cgsc[:,:,np.newaxis]*self.X_t[:,np.newaxis,:],axis=0)+self.shape #[C,D]
#         self.hp_scale_cgsc = 1/(1/self.scale+np.sum(self.y_cgsc,axis=0)[:,np.newaxis]) #[C,D]
#         #<<<
        
#         self.Collapsed_cycle()
    
    
    def Collapsed_cycle(self,need_elbo=False):
        
        pass
#         '''
        
#         run one collapsed Gibbs sampling cycle
        
#         ===== arguments =====
        
#         [1] need_elbo[bool] ... if true, return ELBO, or np.nan
        
#         '''
                
#         self.hp_cent_cgsc = self.hp_cent_cgsc[np.newaxis,:]-self.y_cgsc #[N,C]
#         self.hp_shape_cgsc = self.hp_shape_cgsc[np.newaxis,:,:]-self.y_cgsc[:,:,np.newaxis]*self.X_t[:,np.newaxis,:] #[N,C,D]
#         self.hp_scale_cgsc = 1/(1/self.hp_scale_cgsc[np.newaxis,:,:]-self.y_cgsc[:,:,np.newaxis]) #[N,C,D]

#         nb_p = 1/(1+1/self.hp_scale_cgsc) #[N,C,D]
#         nb_x = spsp.binom(self.X_t[:,np.newaxis,:]+self.hp_shape_cgsc-1,self.hp_shape_cgsc-1)*\
#                     ((1-nb_p)**self.hp_shape_cgsc)*nb_p**self.X_t[:,np.newaxis,:]

#         self.hp_pi_cgsc = self.hp_cent_cgsc*np.prod(nb_x,axis=2) #[N,C]
            
#         self.y_cgsc = np.zeros((self.N,self.C),dtype=int) #[N,C]
#         Wei = np.ones(self.N,dtype=int) #Wei[N]
        
#         for c in range(self.C-1):
#             prb = self.hp_pi_cgsc[:,c]/np.sum(self.hp_pi_cgsc[:,c:],axis=1)
#             prb[np.isnan(prb)] = 0
#             self.y_cgsc[:,c] = np.random.binomial(Wei,prb)
#             Wei-=self.y_cgsc[:,c]
#         self.y_cgsc[:,self.C-1] = Wei
        
#         self.hp_cent_cgsc = np.sum(self.y_cgsc,axis=0)+self.cent #[C]
#         self.hp_shape_cgsc = np.sum(self.y_cgsc[:,:,np.newaxis]*self.X_t[:,np.newaxis,:],axis=0)+self.shape #[C,D]
#         self.hp_scale_cgsc = 1/(1/self.scale+np.sum(self.y_cgsc,axis=0)[:,np.newaxis]) #[C,D]
        
#         if need_elbo:
#             return self.get_elbo('cgs')
#         else:
#             return np.nan
        
    
    def CollapsedGibbsSampling(self,ITER=500,burnin=None,need_elbo=False,seed=None):
        
        pass
#         '''
        
#         run collapsed Gibbs sampling
        
#         ===== arguments =====
        
#         [1] ITER[int] ... the number of times to run CGS for sampling
        
#         [2] burnin[int] ... the number of times to run CGS for stabilizing sample
        
#         [3] need_elbo[bool] ... whether return ELBO or not
        
#         [4] seed[int or None] ... random seed to run collapsed Gibbs sampling
        
        
#         '''
        
#         if seed:
#             np.random.seed(seed)
        
#         if burnin==None:
#             burnin = self.N
        
#         print('burn-in...')
#         for k in tqdm(range(burnin)):
#             self.Collapsed_cycle()
        
#         self.y_CGS = np.zeros((ITER,self.N,self.C)) #[I,N,C]
#         self.elbo_CGS_trace = np.zeros(ITER) #[I]
#         print('sampling...')
#         for k in tqdm(range(ITER)):
#             self.elbo_CGS_trace[k] = self.Collapsed_cycle(need_elbo)
#             self.y_CGS[k,:,:] = self.y_cgsc

    
    def view_Collapsed_assignment(self,save=False):
        
        pass
#         '''
        
#         visualize CGS estimated label assignment on data
        
#         ===== arguments =====
        
#         [1] save[bool] ... whether save the graph or not
        
#         '''
        
#         fig,axes = plt.subplots(1,self.D,figsize=(6*self.D,6))
#         fig.suptitle('component assignment')
#         for d in range(self.D):
#             x = self.X_t[:,d]
#             y = self.y_cgsc
#             xlab = np.unique(x)
#             ydegree = np.add.accumulate(np.sum((y[:,np.newaxis,:]*(x[:,np.newaxis]==xlab)[:,:,np.newaxis]),axis=0),axis=1)
#             for c in range(self.C):
#                 axes[d].bar(xlab,ydegree[:,self.C-c-1],linewidth=0,label=f'component {c}')
#                 axes[d].set(ylabel='degree',xlabel=f'x_{d}')
#                 if d==0:
#                     axes[d].legend()
#         if save:
#             fig.savefig('transition_VI_'+re.sub('[ :.-]','',str(datetime.datetime.today()))+'.pdf')
        
    


# In[4]:


# ipv = np.ones(2)/2
# tpm = np.array([[0.95,0.05],[0.05,0.95]])
# tens = np.array([[6,],[1,]])
# seed = 2022
# N = 10000
# phm = Poisson_Hidden_Markov_generator(ipv,tpm,tens)
# xdata = phm.generate(N,seed)

# phmm = Poisson_Hidden_Markov(xdata,2)

# phmm.cfVI(ITER=30)

# phmm.view_cfVI_trace()

# phmm.sVI(ITER=30)

# phmm.view_sVI_trace()


# In[ ]:





# In[ ]:




