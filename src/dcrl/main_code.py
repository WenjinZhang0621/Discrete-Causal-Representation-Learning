#!/usr/bin/env python
# coding: utf-8
import numpy as np
import math
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp
from scipy.linalg import svd,inv
from scipy.special import logit, expit  # expit is the logistic function
from itertools import product, combinations
import time
import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
import graphviz
import pydot
import networkx as nx
import pandas as pd
from causallearn.search.ScoreBased.GES import ges
import matplotlib.image as mpimg
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.SHD import SHD
#from pcalg import pc
#from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt
from scipy.stats import norm
import io
import random
from factor_analyzer import Rotator
import concurrent.futures
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import json
from filelock import FileLock
import argparse
from collections import Counter


parser=argparse.ArgumentParser()
parser.add_argument("--start",type=int,required=True)
parser.add_argument("--end",type=int,required=True)
parser.add_argument("--n",type=int,required=True)
parser.add_argument("--k",type=int,required=True)
parser.add_argument("--dag_type",type=str,required=True)
parser.add_argument("--distribution",type=str,required=True)
args=parser.parse_args()

def binary(x,k=None):
    x=np.array(x).reshape(-1,1)
    if k is None:
        k=int(np.max(np.floor(np.log2(x))+1)) if np.all(x>0) else 1
    else:
        kmax=int(np.max(np.floor(np.log2(np.max(x)))+1)) if np.max(x)>0 else 1
        assert k>=kmax,"k must be greater than or equal to the maximum binary length of x"
    divs=np.floor(x/(2**np.arange(k-1,-1,-1))).astype(int)
    r=divs-np.hstack((np.zeros((len(x),1)),2*divs[:,:-1]))
    return r

def TLP(x,tau):
    res=np.sum(np.minimum(np.abs(x),tau))
    return res

def thres(X,tau):
    X_th=X.copy()
    index=np.abs(X)<tau
    X_th[index]=0
    return X_th

def nchoosek_prac(n,k):
    numerator=np.sum(np.log(np.arange(n-k+1,n+1)))
    denominator=np.sum(np.log(np.arange(1,k+1)))
    y=np.exp(numerator-denominator)
    return y

def initialize_function():
    def zero_function(x):
        return 0#torch.zeros_like(x, dtype=torch.float32, requires_grad=True)
    return zero_function

def logit(p):
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



class Generate_Data:
    def __init__(self,N,J,K,Q_type,DAG_type,distribution,lob,upb,lob2,upb2,lob3,upb3,lob4,upb4,lob5,upb5):
        self.N=N  
        self.K=K  
        self.J=J
        self.Q_type=Q_type
        self.DAG_type=DAG_type  #Model type('Markov','Tree',etc.)
        self.distribution=distribution  #Distribution for X('Poisson','Normal',etc.)
        self.upb=upb
        self.lob=lob
        self.lob2=lob2
        self.upb2=upb2
        self.upb3=upb3
        self.lob3=lob3
        self.lob4=lob4
        self.upb4=upb4
        self.lob5=lob5
        self.upb5=upb5
        self.Q=self.generate_Q()
        self.B=self.generate_B() 
        self.gamma=self.generate_gamma()
        self.A=None
        self.X=None

    def generate_Q(self):
        Q=np.vstack((np.eye(self.K),np.eye(self.K),np.eye(self.K)))
        for k in range(self.K-1):
            Q[k,k+1]=1
            Q[k+1,k]=1
        if self.Q_type=='2':
            for k in range(self.K-2):
                Q[k,k+2]=1
                Q[k+2,k]=1
        return Q

    def generate_B(self):
        if self.distribution=='Poisson':
            g=1*np.ones(self.J)
            c=3*np.ones(self.J)  
            beta_true=np.zeros((self.J,self.K + 1))
            beta_part_in=np.zeros((self.J,self.K))
            for j in range(self.J):
                beta_true[j,0]=g[j]
                beta_part_in[j,self.Q[j,:]==1]=(c[j]-g[j])/np.sum(self.Q[j,:]==1)
            beta_true[:,1:]=beta_part_in
        elif self.distribution=='Lognormal':
            g=-1*np.ones(self.J)
            c=2*np.ones(self.J)
            beta_true=np.zeros((self.J,self.K+1))
            beta_part_in=np.zeros((self.J,self.K))
            for j in range(self.J):
                beta_true[j,0]=g[j]
                beta_part_in[j,self.Q[j, :]==1]=(c[j]-g[j])/np.sum(self.Q[j,:]==1)
            beta_true[:,1:]=beta_part_in
        elif self.distribution=='Bernoulli':
            g=-1*np.ones(self.J)
            c=2*np.ones(self.J)
            beta_true=np.zeros((self.J,self.K+1))
            beta_part_in=np.zeros((self.J,self.K))
            for j in range(self.J):
                beta_true[j,0]=g[j]
                beta_part_in[j,self.Q[j, :]==1]=(c[j]-g[j])/np.sum(self.Q[j,:]==1)
            beta_true[:,1:]=beta_part_in
        return beta_true

    def generate_gamma(self):
        return np.ones(self.J)

    def generate_latent_data(self,alternate_prob=True):
        if self.DAG_type=='Markov':
            probs=self.generate_prob_Markov()
            self.A=self.generate_Markov_chain(probs)
        elif self.DAG_type=='Tree':
            self.A=self.generate_tree(alternate_prob)
        elif self.DAG_type=='DiverseTree':
            self.A=self.diverse_tree()
        elif self.DAG_type=='Model-16':
            self.A=self.model16()
        elif self.DAG_type=='Model-8':
            self.A=self.model8()
        elif self.DAG_type=='Model-13':
            self.A=self.model13()
        elif self.DAG_type=='Model-7':
            self.A=self.model7()
        return self.A

    def generate_data(self):
        if self.distribution=='Poisson':
            mu_correct=np.dot(np.column_stack(((np.ones(self.N)).reshape(-1,1),self.A)),self.B.T)
            self.X=np.random.poisson(mu_correct)
        elif self.distribution=='Lognormal':
            mu_correct=np.dot(np.column_stack(((np.ones(self.N)).reshape(-1,1),self.A)),self.B.T)
            s=np.tile(np.sqrt(self.gamma),(self.N,1))
            self.X=np.exp(mu_correct+np.random.normal(0,s,(self.N,self.J)))
        elif self.distribution=='Bernoulli':
            mu_correct=np.dot(np.column_stack(((np.ones(self.N)).reshape(-1,1),self.A)),self.B.T)
            self.X= np.random.binomial(1, sigmoid(mu_correct))
        return self.X

    def generate_Markov_chain(self,probs):
        chain=np.zeros((self.N,self.K))
        chain[:,0]=np.random.rand(self.N)<probs[0]
        for k in range(1,self.K):
            prev_state=chain[:,k-1]
            p1,p0=probs[k]
            chain[:,k]=np.where(prev_state==1,np.random.rand(self.N)<p1,np.random.rand(self.N)<p0)
        return chain

    def generate_prob_Markov(self):
        result=[0.5]  
        a=self.lob
        b=self.upb
        for _ in range(self.K-1):  
            c=random.random()
            if c>0.5:
                x=random.uniform(a,b)
                y=random.uniform(1-b,1-a)
            else:
                y=random.uniform(a,b)
                x=random.uniform(1-b,1-a)
            result.append((x, y))
        return result

    def generate_tree(self,alternate_prob):
        tree=np.zeros((self.N,self.K),dtype=int)  
        tree[:,0]=np.random.rand(self.N)<0.5
        node_index=1  
        parent_indices=[0]  
        a=self.lob
        b=self.upb
        while node_index<self.K:
            next_parent_indices=[] 
            for parent_idx in parent_indices:
                if node_index>=self.K: 
                    break
                level=int(np.log2(parent_idx+1))+1  
                if alternate_prob:
                    if level%2==1: 
                        p1,p0=random.uniform(a,b),random.uniform(1-b,1-a)
                    else:            
                        p1,p0=random.uniform(1-b,1-a),random.uniform(a,b)
                else:
                    p1,p0=random.uniform(a,b),random.uniform(1-b,1-a)  
                for _ in range(2):
                    if node_index>=self.K:  
                        break
                    parent_state=tree[:,parent_idx]
                    p=np.where(parent_state==1,p1,p0)
                    tree[:,node_index]=np.random.rand(self.N)<p
                    next_parent_indices.append(node_index)
                    node_index+=1
            parent_indices=next_parent_indices
        return tree

    def diverse_tree(self):
        p0,p1,p2,p3,p4=random.uniform(self.lob,self.upb),random.uniform(self.lob2,self.upb2),random.uniform(self.lob3,self.upb3),random.uniform(self.lob4,self.upb4),random.uniform(self.lob5,self.upb5)
        diverse_btree=DiverseBTree(self.N,self.K,p4,p0,p1, p2, p3)
        tree_structure,chain_structure,matrix=diverse_btree.build()
        return matrix
    
    def model8(self):
        p0,p1,p2,p3,p4=random.uniform(self.lob,self.upb),random.uniform(self.lob2,self.upb2),random.uniform(self.lob3,self.upb3),random.uniform(self.lob4,self.upb4),random.uniform(self.lob5,self.upb5)
        matrix=np.zeros((self.N,self.K))
        for i in range(self.N):
            matrix[i,0]= np.random.rand()<0.5
            matrix[i,1],matrix[i,2]=(np.random.rand()<p4 if matrix[i,0]==1 else np.random.rand()<1-p4,
                                     np.random.rand()<1-p4 if matrix[i,0]==1 else np.random.rand()<p4)
            matrix[i,3],matrix[i,4]=(np.random.rand()<p4 if matrix[i,1]==1 else np.random.rand()<1-p4,
                                    np.random.rand()<1-p4 if matrix[i,1]==1 else np.random.rand()<p4)
            matrix[i,5],matrix[i,6]=(np.random.rand()<p4 if matrix[i,2]==1 else np.random.rand()<1-p4,
                                    np.random.rand()<1-p4 if matrix[i,2]==1 else np.random.rand()<p4)
            matrix[i,7]=(np.random.rand()<p0 if matrix[i,4]==1 and matrix[i,6]==1 else 
            np.random.rand()<p1 if matrix[i,4]==1 and matrix[i,6]==0 else 
            np.random.rand()<p2 if matrix[i,4]==0 and matrix[i,6]==1 else 
            np.random.rand()<p3)
        return matrix

    def model7(self):
        p0,p1,p2,p3,p4=random.uniform(self.lob,self.upb),random.uniform(self.lob2,self.upb2),random.uniform(self.lob3,self.upb3),random.uniform(self.lob4,self.upb4),random.uniform(self.lob5,self.upb5)
        matrix=np.zeros((self.N,self.K))
        for i in range(self.N):
            matrix[i,0]= np.random.rand()<0.5
            matrix[i,1],matrix[i,2]=(np.random.rand()<p4 if matrix[i,0]==1 else np.random.rand()<1-p4,
                                     np.random.rand()<1-p4 if matrix[i,0]==1 else np.random.rand()<p4)
            matrix[i,3],matrix[i,4]=(np.random.rand()<p0 if matrix[i,1]==1 and matrix[i,2]==1 else 
            np.random.rand()<p1 if matrix[i,1]==1 and matrix[i,2]==0 else 
            np.random.rand()<p2 if matrix[i,1]==0 and matrix[i,2]==1 else 
            np.random.rand()<p3,
            np.random.rand()<p0 if matrix[i,1]==1 and matrix[i,2]==1 else 
            np.random.rand()<p1 if matrix[i,1]==1 and matrix[i,2]==0 else 
            np.random.rand()<p2 if matrix[i,1]==0 and matrix[i,2]==1 else 
            np.random.rand()<p3)
            matrix[i,5],matrix[i,6]=(np.random.rand()<p4 if matrix[i,3]==1 else np.random.rand()<1-p4,
                                    np.random.rand()<1-p4 if matrix[i,4]==1 else np.random.rand()<p4)
        return matrix
    
    def model13(self):
        p0,p1,p2,p3,p4=random.uniform(self.lob,self.upb),random.uniform(self.lob2,self.upb2),random.uniform(self.lob3,self.upb3),random.uniform(self.lob4,self.upb4),random.uniform(self.lob5,self.upb5)
        matrix=np.zeros((self.N,self.K))
        for i in range(self.N):
            matrix[i,0]= np.random.rand()<0.5
            matrix[i,1],matrix[i,2]=(np.random.rand()<p4 if matrix[i,0]==1 else np.random.rand()<1-p4,
                                     np.random.rand()<1-p4 if matrix[i,0]==1 else np.random.rand()<p4)
            matrix[i,3],matrix[i,4]=(np.random.rand()<p4 if matrix[i,1]==1 else np.random.rand()<1-p4,
                                    np.random.rand()<1-p4 if matrix[i,1]==1 else np.random.rand()<p4)
            matrix[i,5],matrix[i,6]=(np.random.rand()<p4 if matrix[i,2]==1 else np.random.rand()<1-p4,
                                    np.random.rand()<1-p4 if matrix[i,2]==1 else np.random.rand()<p4)
            matrix[i,7],matrix[i,8],matrix[i,9]=(
            np.random.rand()<p0 if matrix[i,3]==1 and matrix[i,4]==1 else 
            np.random.rand()<p1 if matrix[i,3]==1 and matrix[i,4]==0 else 
            np.random.rand()<p2 if matrix[i,3]==0 and matrix[i,4]==1 else 
            np.random.rand()<p3,
            np.random.rand()<p0 if matrix[i,3]==1 and matrix[i,4]==1 else 
            np.random.rand()<p1 if matrix[i,3]==1 and matrix[i,4]==0 else 
            np.random.rand()<p2 if matrix[i,3]==0 and matrix[i,4]==1 else 
            np.random.rand()<p3,
            np.random.rand()<p0 if matrix[i,3]==1 and matrix[i,4]==1 else 
            np.random.rand()<p1 if matrix[i,3]==1 and matrix[i,4]==0 else 
            np.random.rand()<p2 if matrix[i,3]==0 and matrix[i,4]==1 else 
            np.random.rand()<p3)
            matrix[i,10],matrix[i,11],matrix[i,12]=(
            np.random.rand()<p0 if matrix[i,5]==1 and matrix[i,6]==1 else 
            np.random.rand()<p1 if matrix[i,5]==1 and matrix[i,6]==0 else 
            np.random.rand()<p2 if matrix[i,5]==0 and matrix[i,6]==1 else 
            np.random.rand()<p3,
            np.random.rand()<p0 if matrix[i,5]==1 and matrix[i,6]==1 else 
            np.random.rand()<p1 if matrix[i,5]==1 and matrix[i,6]==0 else 
            np.random.rand()<p2 if matrix[i,5]==0 and matrix[i,6]==1 else 
            np.random.rand()<p3,
            np.random.rand()<p0 if matrix[i,5]==1 and matrix[i,6]==1 else 
            np.random.rand()<p1 if matrix[i,5]==1 and matrix[i,6]==0 else 
            np.random.rand()<p2 if matrix[i,5]==0 and matrix[i,6]==1 else 
            np.random.rand()<p3)
        return matrix

    def model16(self):
        p0,p1,p2,p3,p4=random.uniform(self.lob,self.upb),random.uniform(self.lob2,self.upb2),random.uniform(self.lob3,self.upb3),random.uniform(self.lob4,self.upb4),random.uniform(self.lob5,self.upb5)
        matrix=np.zeros((self.N,self.K))
        for i in range(self.N):
            matrix[i,0]= np.random.rand()<0.5
            matrix[i,1],matrix[i,2]=(np.random.rand()<p4 if matrix[i,0]==1 else np.random.rand()<1-p4,
                                     np.random.rand()<1-p4 if matrix[i,0]==1 else np.random.rand()<p4)
            matrix[i,3]=(np.random.rand()<p0 if matrix[i,1]==1 and matrix[i,2]==1 else 
            np.random.rand()<p1 if matrix[i,1]==1 and matrix[i,2]==0 else 
            np.random.rand()<p2 if matrix[i,1]==0 and matrix[i,2]==1 else 
            np.random.rand()<p3)
            matrix[i,4],matrix[i,5]=(np.random.rand()<p4 if matrix[i,3]==1 else np.random.rand()<1-p4,
                                    np.random.rand()<1-p4 if matrix[i,3]==1 else np.random.rand()<p4)
            matrix[i,6],matrix[i,7],matrix[i,8],matrix[i,9]=(np.random.rand()<p4 if matrix[i,4]==1 else np.random.rand()<1-p4,
                                                            np.random.rand()<1-p4 if matrix[i,4]==1 else np.random.rand()<p4,
                                                            np.random.rand()<p4 if matrix[i,5]==1 else np.random.rand()<1-p4,
                                                            np.random.rand()<1-p4 if matrix[i,5]==1 else np.random.rand()<p4)
            matrix[i,10],matrix[i,11],matrix[i,12]=(
            np.random.rand()<p0 if matrix[i,6]==1 and matrix[i,7]==1 else 
            np.random.rand()<p1 if matrix[i,6]==1 and matrix[i,7]==0 else 
            np.random.rand()<p2 if matrix[i,6]==0 and matrix[i,7]==1 else 
            np.random.rand()<p3,
            np.random.rand()<p0 if matrix[i,6]==1 and matrix[i,7]==1 else 
            np.random.rand()<p1 if matrix[i,6]==1 and matrix[i,7]==0 else 
            np.random.rand()<p2 if matrix[i,6]==0 and matrix[i,7]==1 else 
            np.random.rand()<p3,
            np.random.rand()<p0 if matrix[i,6]==1 and matrix[i,7]==1 else 
            np.random.rand()<p1 if matrix[i,6]==1 and matrix[i,7]==0 else 
            np.random.rand()<p2 if matrix[i,6]==0 and matrix[i,7]==1 else 
            np.random.rand()<p3)
            matrix[i,13],matrix[i,14],matrix[i,15]=(
            np.random.rand()<p0 if matrix[i,8]==1 and matrix[i,9]==1 else 
            np.random.rand()<p1 if matrix[i,8]==1 and matrix[i,9]==0 else 
            np.random.rand()<p2 if matrix[i,8]==0 and matrix[i,9]==1 else 
            np.random.rand()<p3,
            np.random.rand()<p0 if matrix[i,8]==1 and matrix[i,9]==1 else 
            np.random.rand()<p1 if matrix[i,8]==1 and matrix[i,9]==0 else 
            np.random.rand()<p2 if matrix[i,8]==0 and matrix[i,9]==1 else 
            np.random.rand()<p3,
            np.random.rand()<p0 if matrix[i,8]==1 and matrix[i,9]==1 else 
            np.random.rand()<p1 if matrix[i,8]==1 and matrix[i,9]==0 else 
            np.random.rand()<p2 if matrix[i,8]==0 and matrix[i,9]==1 else 
            np.random.rand()<p3)
        return matrix


# In[ ]:


class DiverseBTree:
    def __init__(self,N,K,p0,p1,p2,p3,p4):
        self.K=K 
        self.p0=p0  
        self.p1=p1
        self.p2=p2  
        self.p3=p3  
        self.p4=p4  
        self.tree_structure=[] 
        self.chain_structure=[]
        self.N=N  
        self.matrix=None

    def _generate_tree_structure(self):
        nearest_power_of_two_minus_1 = 2**(int(np.log2(self.K + 1))) - 1
        remaining_nodes = nearest_power_of_two_minus_1
        layers=[]
        while remaining_nodes > 0:
            power_of_two = 2 ** (int(np.log2(remaining_nodes)))
            layers.append(power_of_two)
            remaining_nodes -= power_of_two
        self.tree_structure = layers

    def _generate_chain_structure(self):
        remaining_nodes = self.K - (2**(int(np.log2(self.K + 1))) - 1)
        self.chain_structure = [i for i in range(self.K - remaining_nodes, self.K)]

    def _generate_matrix(self):
        matrix = np.zeros((self.N, self.K))
        matrix[:,range(self.tree_structure[0])]=(np.random.rand(self.N,self.tree_structure[0])<0.5)
        ter_ind=0
        for level in range(1,len(self.tree_structure)):
                num_nodes=self.tree_structure[level]
                new_matrix=np.zeros((self.N,num_nodes))
                parent=[i for i in range(ter_ind,ter_ind+2*num_nodes)]
                child=[i for i in range(ter_ind+2*num_nodes,ter_ind+3*num_nodes)]
                focus=matrix[:,parent]
                for node in range(num_nodes):
                    col1,col2=2*node,2*node+1
                    new_matrix[:,node]=np.where((focus[:,col1]==1)&(focus[:,col2]==1),self.p0,
                        np.where((focus[:,col1]==1)&(focus[:,col2]==0),self.p1,
                        np.where((focus[:,col1]==0)&(focus[:,col2]==1),self.p2,self.p3)))
                    matrix[:,child]=(np.random.rand(self.N,num_nodes)<new_matrix)
                ter_ind=ter_ind+2*num_nodes
        for k in range(ter_ind+1,self.K):
            prev_state=matrix[:,k-1]
            matrix[:,k]=np.where(prev_state==1,np.random.rand(self.N)<1-self.p4,np.random.rand(self.N)<self.p4)
        self.matrix = matrix

    def build(self):
        self._generate_tree_structure()
        self._generate_chain_structure()
        self._generate_matrix()
        return self.tree_structure, self.chain_structure, self.matrix


# In[9]:


class DAG_Estimator(Generate_Data):
    def __init__(self,N,J,K,Q_type,DAG_type,distribution,algorithm,upb,lob,lob2,upb2,lob3,upb3,lob4,upb4,lob5,upb5,tau,pen,max_iter,tol,C,epsilon,Q_N,kappa,L_final):
        super().__init__(N,J,K,Q_type,DAG_type,distribution,upb,lob,lob2,upb2,lob3,upb3,lob4,upb4,lob5,upb5)
        #self.p_hat=p_ini
        #self.B_hat=B_ini
        #self.gamma_hat=gamma_ini
        #self.A_hat=A_ini
        self.tau=tau
        self.pen=pen
        self.max_iter=max_iter
        self.tol=tol    
        self.C=C
        self.epsilon=epsilon
        self.p_hat=None
        self.B_hat=None
        self.gamma_hat=None
        self.A_hat=None
        self.Q_N=Q_N
        self.algorithm=algorithm
        self.kappa=kappa
        self.L_final=L_final

    def ftn_T(self,X):
        if self.distribution=='Poisson':
            T=np.atleast_2d(np.array(X))
        elif self.distribution=='Lognormal':#(1,y)\to(2,y)
            T_1=-(np.log(X))**2
            T_2=np.log(X)
            T=np.vstack((T_1,T_2))
        elif self.distribution=='Bernoulli':
            T=np.atleast_2d(np.array(X))
        return T

    def ftn_h(self,Y):
        if self.distribution=='Poisson':#(1,y)\to(1,y)
            C=np.atleast_2d(np.log(Y[0,:]))
        elif self.distribution=='Lognormal':#(2,y)\to(2,y)
            Y_1=Y[0,:]
            Y_2=np.maximum(Y[1,:],1e-100)
            C=np.vstack((1/(2*Y_2),Y_1/Y_2))#1/2\sigma^2,\mu/\sigma^2
        elif self.distribution=='Bernoulli':#(1,y)\to(1,y)
            C=np.atleast_2d(Y[0, :])
        return C

    def ftn_A(self,eta):
        if self.distribution=='Poisson':
            C=np.exp(eta)
        elif self.distribution=='Lognormal':#(J,2)\to(J,1)
            eta_1=np.maximum(eta[:,0],1e-100)
            eta_2=eta[:,1]
            C=(eta_2**2/(4*eta_1)+np.log(1/(2*eta_1))/2).reshape(len(eta_1),1)
        elif self.distribution=='Bernoulli':
            C=np.log1p(np.exp(eta))
        return C

    def objective(self,phi,j):
        if self.distribution=='Poisson':
            def obj(x):
                beta_long=x.reshape(self.K+1,1)
                tmp=0
                penalty=self.pen*TLP(x,self.tau)
                for a in range(2**self.K):
                    eta=np.log(np.dot((np.insert(binary(a,self.K).flatten(),0,1)).reshape(1,1+self.K),beta_long))
                    tmp+=np.dot(eta.T,self.ftn_T(self.X[:,j]))@phi[:,a]-np.sum(self.ftn_A(eta.T))*np.sum(phi[:,a])
                return penalty-tmp 
        elif self.distribution=='Lognormal':
            def obj(x):
                beta_long=x[:-1].reshape(self.K+1,1)
                gamma=np.array(x[-1]).reshape(1,1)
                tmp=0
                penalty=self.pen*TLP(x,self.tau)
                for a in range(2**self.K):
                    eta=self.ftn_h(np.vstack((np.dot((np.insert(binary(a,self.K).flatten(),0,1)).reshape(1,1+self.K),beta_long),gamma)))
                    tmp+=np.dot(eta.T,self.ftn_T(self.X[:, j]))@phi[:,a]-np.sum(self.ftn_A(eta.T))*np.sum(phi[:,a])
                return penalty-tmp 
        elif self.distribution=='Bernoulli':
            def obj(x):
                beta_long=x.reshape(self.K+1,1)
                tmp=0
                penalty=self.pen*TLP(x,self.tau)
                for a in range(2**self.K):
                    eta = np.dot((np.insert(binary(a,self.K).flatten(),0,1)).reshape(1,1+self.K), beta_long)  # (1,1)
                    tmp += np.dot(eta.T, self.ftn_T(self.X[:,j])) @ phi[:,a] - np.sum(self.ftn_A(eta.T)) * np.sum(phi[:,a])
                return penalty-tmp 
        return obj

    def ftn_pen(self,beta_gamma):
        if self.distribution=='Poisson':
            return self.pen*TLP(beta_gamma[1:,],self.tau)
        elif self.distribution=='Lognormal':
            return self.pen*TLP(beta_gamma[1:-1],self.tau)
        elif self.distribution=='Bernoulli':
            return self.pen*TLP(beta_gamma[1:,],self.tau)

    def F_1_SAEM(self,Xj,A_sample_long):
            A_sample_long=np.array(A_sample_long,dtype=np.float32)
            if self.distribution=='Poisson':
                def obj(x):
                    beta_long=x.reshape(self.K+1,1)
                    tmp=0
                    for c in range(self.C):
                        A_beta=np.dot(A_sample_long[:,:,c],beta_long)  # (N, 1)
                        eta=self.ftn_h(A_beta.T)  
                        tmp+=np.sum(eta*self.ftn_T(Xj))-np.sum(self.ftn_A(eta.T))
                    return -(tmp/self.C)
            elif self.distribution=='Lognormal':
                def obj(x):
                    beta_long=x[:-1].reshape(self.K+1,1)
                    gamma=x[-1].reshape(1,1)
                    tmp=0
                    for c in range(self.C):
                        A_beta=np.dot(A_sample_long[:,:,c],beta_long)  # (N, 1)
                        eta=self.ftn_h(np.vstack((A_beta.T,np.tile(gamma,(1,self.N)))))  # (2, N)
                        tmp+=np.sum(eta*self.ftn_T(Xj))-np.sum(self.ftn_A(eta.T))
                    return -(tmp/self.C)
            elif self.distribution=='Bernoulli':
                def obj(x):
                    beta_long=x.reshape(self.K+1,1)
                    tmp=0
                    for c in range(self.C):
                        A_beta=np.dot(A_sample_long[:,:,c],beta_long)  # (N, 1)
                        eta = self.ftn_h(A_beta.T)                                # (1, N)
                        tmp += np.sum(eta * self.ftn_T(Xj)) - np.sum(self.ftn_A(eta.T))
                    return -(tmp/self.C)
            return obj

    def PEM(self,ite):
        self.p_hat,self.B_hat,self.gamma_hat,self.A_hat=self.init(ite)
        Record=ges(self.A_hat,score_func='local_score_BDeu')
        pyd=GraphUtils.to_pydot(Record['G'])
        pyd.write_png(f'A_ini_{self.N}_ges_BDeu_{self.distribution}_{self.DAG_type}_{ite}.png') 
        n_in=2**self.K
        err=1
        itera=0
        loglik=0
        options={'disp':False}
        if self.distribution=='Poisson':
            lb=[(np.zeros(self.K+1)) for j in range(self.J)]
            ub=[np.concatenate(([2],3*np.ones(self.K))) for j in range(self.J)]
            while abs(err)>self.tol and itera<self.max_iter:
        #E-step
                old_loglik=loglik
                phi=np.zeros((self.N,n_in))
                psi=np.zeros(n_in).reshape(n_in,1)
                exponent=np.zeros(n_in)
                for i in range(self.N):
                    for a in range(n_in):
                        eta=self.ftn_h(np.dot((np.insert(binary(a,self.K).flatten(),0,1)).reshape(1,1+self.K),self.B_hat.T)).T
                        exponent[a]=np.sum(np.matrix(np.diag(eta@self.ftn_T(self.X[i,:]))).T-self.ftn_A(eta))
                    logphi_i=exponent.reshape(n_in,1)+np.log(self.p_hat)
                    log_max=np.max(logphi_i)
                    exp_shifted=np.exp(logphi_i-log_max)
                    phi[i,:]=exp_shifted.flatten()/np.sum(exp_shifted)
        #M-step
                for a in range(n_in):
                    psi[a]=np.sum(phi[:,a])
                self.p_hat=psi/np.sum(psi)
                for j in range(self.J):
                    f=self.objective(phi,j)
                    opt_result=minimize(f,self.B_hat[j,:].flatten(),bounds=list(zip(lb[j],ub[j])),options=options)
                    self.B_hat[j,:]=opt_result.x
                tmp=0
                for i in range(self.N):
                    for a in range(n_in):
                        eta=self.ftn_h(np.dot((np.insert(binary(a,self.K).flatten(),0,1)).reshape(1,1+self.K),self.B_hat.T)).T
                        exponent[a]=np.sum(np.diag(eta@self.ftn_T(self.X[i,:]))-self.ftn_A(eta))
                    tmp+=np.log(np.dot(self.p_hat.T,np.exp(exponent)))
                loglik=tmp
                err=loglik-old_loglik
                itera+=1
                print('EM Iteration {}, Err {}'.format(itera, err))
            return self.p_hat,self.B_hat,loglik,itera
        elif self.distribution=='Lognormal':
            lb=[np.concatenate(([-2],np.zeros(self.K+1))) for j in range(self.J)]
            ub=[np.concatenate((4*np.ones(self.K+1),[2])) for j in range(self.J)]
            while abs(err)>self.tol and itera<self.max_iter:
        #E-step
                old_loglik=loglik
                phi=np.zeros((self.N,n_in))
                psi=np.zeros(n_in).reshape(n_in,1)
                exponent=np.zeros(n_in)
                for i in range(self.N):
                    for a in range(n_in):
                        eta=self.ftn_h(np.vstack((np.dot((np.insert(binary(a,self.K).flatten(),0,1)).reshape(1,1+self.K),self.B_hat.T),self.gamma_hat.T))).T#(J,2)
                        exponent[a]=np.sum(np.matrix(np.diag(eta@self.ftn_T(self.X[i,:]))).T-self.ftn_A(eta)) 
                    logphi_i=exponent.reshape(n_in,1)+np.log(self.p_hat)
                    log_max=np.max(logphi_i)
                    exp_shifted=np.exp(logphi_i-log_max)
                    phi[i,:]=exp_shifted.flatten()/np.sum(exp_shifted)
        #M-step
                for a in range(n_in):
                    psi[a]=np.sum(phi[:,a])
                self.p_hat=psi/np.sum(psi)
                for j in range(self.J):
                    f=self.objective(phi,j)
                    opt_result=minimize(f,np.concatenate((self.B_hat[j,:].flatten(),np.array(self.gamma_hat[j]))),bounds=list(zip(lb[j],ub[j])),options=options)
                    opt=opt_result.x
                    self.B_hat[j,:]=opt[:-1]
                    self.gamma_hat[j]=opt[-1]
                tmp=0
                for i in range(self.N):
                    for a in range(n_in):
                        eta=self.ftn_h(np.vstack((np.dot((np.insert(binary(a,self.K).flatten(),0,1)).reshape(1,1+self.K),self.B_hat.T),self.gamma_hat.T))).T
                        exponent[a]=np.sum(np.diag(eta@self.ftn_T(self.X[i,:]))-self.ftn_A(eta))
                    tmp+=np.log(np.dot(self.p_hat.T,np.exp(exponent)))
                loglik=tmp
                err=loglik-old_loglik
                itera+=1
                print('EM Iteration {}, Err {}'.format(itera, err))
            return self.p_hat,self.B_hat,self.gamma_hat.reshape(-1,1),loglik,itera

    def PSAEM(self,ite):
        self.p_hat,self.B_hat,self.gamma_hat,self.A_hat=self.init(ite)
        #Record=ges(self.A_hat,score_func='local_score_BDeu')
        #pyd=GraphUtils.to_pydot(Record['G'])
        #pyd.write_png(f'Q_A_ini_{self.N}_ges_BDeu_{self.distribution}_{self.DAG_type}_{ite}.png') 
        pow2 = (1 << np.arange(self.K-1, -1, -1))
        err=1
        t=0
        loglik=0
        rows_idx0 = (self.A_hat * pow2).sum(axis=1).astype(np.int64)
        counts_smooth = Counter()
        for u, c in zip(*np.unique(rows_idx0, return_counts=True)):
            counts_smooth[int(u)] = float(c)
        options={'disp': False, 'maxiter': self.max_iter}
        if self.distribution=='Poisson':
            lb =(np.zeros(self.K+1))
            ub = np.concatenate(([2],3*np.ones(self.K)))
            B_update=np.zeros((self.J,self.K+1))
            iter_indicator=True
            A_new=self.A_hat.copy()
            A_sample_long=np.zeros((self.N,self.K+1,self.C))
            f_old_1=[initialize_function() for _ in range(self.J)]
        # Iteration start
            #progress_bar=tqdm(total=max_iter,desc="SAEM Iterations")
            while iter_indicator:
                A_cur=A_new.copy()  
        #E-step
                counts_hat = Counter()
                ones_col = np.ones((self.N, 1), dtype=int)  
                for c in range(self.C):
                    idx_cur = (A_cur * pow2).sum(axis=1).astype(np.int64) 
                    for i in range(self.N):
                        z_i = np.dot(np.insert(A_cur[i], 0, 1, axis=0), (self.B_hat).T).astype(float) 
                        T_i = self.ftn_T(self.X[i,:]).T
                        for k in np.random.permutation(self.K):
                            mask = int(pow2[k])                        
                            col  =self.B_hat[:, k+1].astype(float) 
                            if A_cur[i, k] == 1:
                                z0 = z_i - col                          
                                z1 = z_i                                
                                idx1 = idx_cur[i] |  mask
                                idx0 = idx_cur[i] & ~mask
                            else:
                                z0 = z_i                                 
                                z1 = z_i + col                         
                                idx1 = idx_cur[i] |  mask
                                idx0 = idx_cur[i] & ~mask
                            log_prior_ratio = np.log(counts_smooth.get(idx1, 0.0) + self.kappa) - np.log(counts_smooth.get(idx0, 0.0) + self.kappa)
                            eta1=self.ftn_h(np.maximum(z1.reshape(1, -1),1e-150))
                            eta2=self.ftn_h(np.maximum(z0.reshape(1, -1),1e-150))
                            eta1T = eta1.T;  eta2T = eta2.T                            # (J,2)
                            dots1 = np.sum(eta1T * T_i, axis=1)                    # (J,)
                            dots0 = np.sum(eta2T *T_i, axis=1)                    # (J,)
                            A1 = self.ftn_A(eta1T)[:, 0]                                    # (J,)
                            A0 = self.ftn_A(eta2T)[:, 0]                                    # (J,)
                            pa1 = float(np.sum(dots1 - A1))                            
                            pa0 = float(np.sum(dots0 - A0)) 

                            prob1 = expit(log_prior_ratio + (pa1 - pa0))
                            new_bit = np.random.binomial(1, prob1)
                            if new_bit != A_cur[i, k]:
                                A_cur[i, k] = new_bit
                                z_i = z1 if new_bit == 1 else z0                       # (J,)
                                idx_cur[i] = idx1 if new_bit == 1 else idx0  
                    A_sample_long[:,:,c]=np.hstack((ones_col, A_cur)) 
                    uniq, cnts = np.unique(idx_cur, return_counts=True)
                    for k_idx, v in zip(uniq.tolist(), cnts.tolist()):
                        counts_hat[k_idx] = counts_hat.get(int(k_idx), 0.0) + float(v)
                c, t0 = 0.5, 10.0     
                step = c / (t + t0)
                if self.C > 1:
                    for k_idx in list(counts_hat.keys()):
                        counts_hat[k_idx] /= float(self.C)
                if counts_smooth:
                    for k_idx in list(counts_smooth.keys()):
                        counts_smooth[k_idx] *= (1.0 - step)
                for k_idx, v in counts_hat.items():
                    counts_smooth[k_idx] = counts_smooth.get(k_idx, 0.0) + step * float(v)

        # update effective sample size (kept small and scalar)
                A_new = A_cur.copy()
                # M-step: Update model parameters with stochastic approximation
                for j in range(self.J):
                    Xj=self.X[:,j].copy()  
                    f_loglik=self.F_1_SAEM(Xj,A_sample_long) 
                    def update_f_old_1(x,f_old=f_old_1[j],f_new=f_loglik):
                        return (1-step)*f_old(x)+step*f_new(x)
                    f_old_1[j]=update_f_old_1
                    def f_j(beta_gamma):
                        return f_old_1[j](beta_gamma)+self.ftn_pen(beta_gamma)
                    #progress_bar = tqdm(total=max_opt_iter, desc=f"Optimizing Parameter {j}")
                    #def callback(xk):
                    #    progress_bar.update(1)
                    opt_result=minimize(f_j,self.B_hat[j, :].flatten(),bounds=list(zip(lb,ub)),method='SLSQP',options=options)
                    #progress_bar.close()
                    B_update[j,:]=opt_result.x
                err=np.linalg.norm(self.B_hat-np.concatenate((B_update[:,[0]],thres(B_update[:,1:],self.tau)),axis=1),'fro')**2
                self.B_hat=np.concatenate((B_update[:,[0]],thres(B_update[:,1:],self.tau)),axis=1)
                t+=1
                #progress_bar.update(1)
                print(f'SAEM Iteration {t}, Err {err:.5f}')
                iter_indicator=(abs(err)>self.tol and t<self.max_iter)   
        
            U_T = (1 << self.K) - len(counts_smooth)
            nu_dense = np.zeros(1 << self.K, dtype=float)

            if counts_smooth:
                idxs = np.fromiter(counts_smooth.keys(), dtype=np.int64)
                vals = np.fromiter((counts_smooth[k] for k in idxs), dtype=float)
                nu_dense[idxs] = vals

            if self.L_final > 0.0 and U_T > 0:
                nu_dense += (self.L_final / U_T)

            total = nu_dense.sum()
            if total > 0:
                nu_dense /= total
            else:
                nu_dense.fill(1.0 / (1 << self.K))

            self.p_hat = nu_dense.reshape(-1, 1)
            #progress_bar.close()
            return self.p_hat,self.B_hat,self.A_hat,t,loglik            
        elif self.distribution=='Lognormal':
            lb=np.concatenate(([-2],np.zeros(self.K+1)))
            ub=np.concatenate((4*np.ones(self.K+1),[2]))
            B_update=np.zeros((self.J,self.K+1))
            gamma_update=np.zeros((self.J,1))
            iter_indicator=True
            A_new=self.A_hat.copy()
            A_sample_long=np.zeros((self.N,self.K+1,self.C))
            f_old_1=[initialize_function() for _ in range(self.J)]
        # Iteration start
            #progress_bar=tqdm(total=max_iter,desc="SAEM Iterations")
            while iter_indicator:
                A_cur=A_new.copy()  
        #E-step
                counts_hat = Counter()
                ones_col = np.ones((self.N, 1), dtype=int)  
                for c in range(self.C):
                    idx_cur = (A_cur * pow2).sum(axis=1).astype(np.int64) 
                    for i in range(self.N):
                        z_i = np.dot(np.insert(A_cur[i], 0, 1, axis=0), (self.B_hat).T).astype(float) 
                        T_i = self.ftn_T(self.X[i,:]).T
                        for k in np.random.permutation(self.K):
                            mask = int(pow2[k])                        
                            col  =self.B_hat[:, k+1].astype(float) 
                            if A_cur[i, k] == 1:
                                z0 = z_i - col                          # 把该位当 0
                                z1 = z_i                                 # 当前就是 1
                                idx1 = idx_cur[i] |  mask
                                idx0 = idx_cur[i] & ~mask
                            else:
                                z0 = z_i                                 # 当前就是 0
                                z1 = z_i + col                          # 把该位当 1
                                idx1 = idx_cur[i] |  mask
                                idx0 = idx_cur[i] & ~mask
                            log_prior_ratio = np.log(counts_smooth.get(idx1, 0.0) + self.kappa) - np.log(counts_smooth.get(idx0, 0.0) + self.kappa)
                            eta1=self.ftn_h(np.vstack((z1.reshape(1, -1),(self.gamma_hat).reshape(1,-1))))
                            eta2=self.ftn_h(np.vstack((z0.reshape(1, -1),(self.gamma_hat).reshape(1,-1))))
                            eta1T = eta1.T;  eta2T = eta2.T                            # (J,2)
                            dots1 = np.sum(eta1T * T_i, axis=1)                    # (J,)
                            dots0 = np.sum(eta2T *T_i, axis=1)                    # (J,)
                            A1 = self.ftn_A(eta1T)[:, 0]                                    # (J,)
                            A0 = self.ftn_A(eta2T)[:, 0]                                    # (J,)
                            pa1 = float(np.sum(dots1 - A1))                            # 标量
                            pa0 = float(np.sum(dots0 - A0)) 

                            prob1 = expit(log_prior_ratio + (pa1 - pa0))
                            new_bit = np.random.binomial(1, prob1)
                            if new_bit != A_cur[i, k]:
                                A_cur[i, k] = new_bit
                                z_i = z1 if new_bit == 1 else z0                       # (J,)
                                idx_cur[i] = idx1 if new_bit == 1 else idx0  
                    A_sample_long[:,:,c]=np.hstack((ones_col, A_cur)) 
                    uniq, cnts = np.unique(idx_cur, return_counts=True)
                    for k_idx, v in zip(uniq.tolist(), cnts.tolist()):
                        counts_hat[k_idx] = counts_hat.get(int(k_idx), 0.0) + float(v)
                c, t0 = 0.5, 10.0     
                step = c / (t + t0)
                if self.C > 1:
            # divide by C so the effective batch size is N
                    for k_idx in list(counts_hat.keys()):
                        counts_hat[k_idx] /= float(self.C)
                if counts_smooth:
                    for k_idx in list(counts_smooth.keys()):
                        counts_smooth[k_idx] *= (1.0 - step)
                for k_idx, v in counts_hat.items():
                    counts_smooth[k_idx] = counts_smooth.get(k_idx, 0.0) + step * float(v)

                A_new = A_cur.copy()
        # M-step: 
                for j in range(self.J):
                    Xj=self.X[:,j]  
                    f_loglik=self.F_1_SAEM(Xj,A_sample_long) 
                    def update_f_old_1(x,f_old=f_old_1[j],f_new=f_loglik):
                        return (1-step)*f_old(x)+step*f_new(x)
                    f_old_1[j]=update_f_old_1
                    def f_j(beta_gamma):
                        return f_old_1[j](beta_gamma)+self.ftn_pen(beta_gamma)
                    #progress_bar = tqdm(total=max_opt_iter, desc=f"Optimizing Parameter {j}")
                    #def callback(xk):
                    #    progress_bar.update(1)
                    opt_result=minimize(f_j,np.concatenate((self.B_hat[j,:].flatten(),np.array(self.gamma_hat[j]))),bounds=list(zip(lb,ub)),method='SLSQP',options=options)
                    #progress_bar.close()
                    B_update[j,:]=opt_result.x[:-1]
                    gamma_update[j]=opt_result.x[-1]
                err=np.linalg.norm(self.B_hat-np.concatenate((B_update[:,[0]],thres(B_update[:,1:],self.tau)),axis=1),'fro')**2+np.linalg.norm(self.gamma_hat-gamma_update,'fro')**2
                self.B_hat=np.concatenate((B_update[:,[0]],thres(B_update[:,1:],self.tau)),axis=1)
                self.gamma_hat=gamma_update.copy()
                t+=1
                #progress_bar.update(1)
                print(f'SAEM Iteration {t}, Err {err:.5f}')
                iter_indicator=(abs(err)>self.tol and t<self.max_iter)    
            U_T = (1 << self.K) - len(counts_smooth)
            nu_dense = np.zeros(1 << self.K, dtype=float)

            if counts_smooth:
                idxs = np.fromiter(counts_smooth.keys(), dtype=np.int64)
                vals = np.fromiter((counts_smooth[k] for k in idxs), dtype=float)
                nu_dense[idxs] = vals

            if self.L_final > 0.0 and U_T > 0:
                nu_dense += (self.L_final / U_T)

            total = nu_dense.sum()
            if total > 0:
                nu_dense /= total
            else:
                nu_dense.fill(1.0 / (1 << self.K))

            self.p_hat = nu_dense.reshape(-1, 1)
            #progress_bar.close()
            return self.p_hat,self.B_hat,self.gamma_hat,self.A_hat,t,loglik
        elif self.distribution=='Bernoulli':
            lb=np.concatenate(([-5],-2.5*np.ones(self.K)))
            ub=5*np.ones(self.K+1)
            B_update=np.zeros((self.J,self.K+1))
            iter_indicator=True
            A_new=self.A_hat.copy()
            A_sample_long=np.zeros((self.N,self.K+1,self.C))
            p=np.zeros((self.N,self.K))
            f_old_1=[initialize_function() for _ in range(self.J)]
        # Iteration start
            #progress_bar=tqdm(total=max_iter,desc="SAEM Iterations")
            while iter_indicator:
                A_cur=A_new.copy()  
        #E-step
                counts_hat = Counter()
                ones_col = np.ones((self.N, 1), dtype=int) 
                for c in range(self.C):
                    idx_cur = (A_cur * pow2).sum(axis=1).astype(np.int64) 
                    for i in range(self.N):
                        z_i = np.dot(np.insert(A_cur[i], 0, 1, axis=0), (self.B_hat).T).astype(float) 
                        T_i = self.ftn_T(self.X[i,:]).T
                        for k in np.random.permutation(self.K):
                            mask = int(pow2[k])                        
                            col  =self.B_hat[:, k+1].astype(float) 
                            if A_cur[i, k] == 1:
                                z0 = z_i - col                          
                                z1 = z_i                                
                                idx1 = idx_cur[i] |  mask
                                idx0 = idx_cur[i] & ~mask
                            else:
                                z0 = z_i                                 
                                z1 = z_i + col                         
                                idx1 = idx_cur[i] |  mask
                                idx0 = idx_cur[i] & ~mask
                            log_prior_ratio = np.log(counts_smooth.get(idx1, 0.0) + self.kappa) - np.log(counts_smooth.get(idx0, 0.0) + self.kappa)
                            eta1=self.ftn_h(z1.reshape(1, -1))
                            eta2=self.ftn_h(z0.reshape(1, -1))
                            eta1T = eta1.T;  eta2T = eta2.T                            # (J,2)
                            dots1 = np.sum(eta1T * T_i, axis=1)                    # (J,)
                            dots0 = np.sum(eta2T *T_i, axis=1)                    # (J,)
                            A1 = self.ftn_A(eta1T)[:, 0]                                    # (J,)
                            A0 = self.ftn_A(eta2T)[:, 0]                                    # (J,)
                            pa1 = float(np.sum(dots1 - A1))                            
                            pa0 = float(np.sum(dots0 - A0)) 

                            prob1 = expit(log_prior_ratio + (pa1 - pa0))
                            new_bit = np.random.binomial(1, prob1)
                            if new_bit != A_cur[i, k]:
                                A_cur[i, k] = new_bit
                                z_i = z1 if new_bit == 1 else z0                       # (J,)
                                idx_cur[i] = idx1 if new_bit == 1 else idx0  
                    A_sample_long[:,:,c]=np.hstack((ones_col, A_cur)) 
                    uniq, cnts = np.unique(idx_cur, return_counts=True)
                    for k_idx, v in zip(uniq.tolist(), cnts.tolist()):
                        counts_hat[k_idx] = counts_hat.get(int(k_idx), 0.0) + float(v)
                c, t0 = 0.1, 10     
                step = c / (t + t0)
                if self.C > 1:
            # divide by C so the effective batch size is N
                    for k_idx in list(counts_hat.keys()):
                        counts_hat[k_idx] /= float(self.C)
                if counts_smooth:
                    for k_idx in list(counts_smooth.keys()):
                        counts_smooth[k_idx] *= (1.0 - step)
                for k_idx, v in counts_hat.items():
                    counts_smooth[k_idx] = counts_smooth.get(k_idx, 0.0) + step * float(v)

                A_new = A_cur.copy()         
        # M-step: Update model parameters with stochastic approximation
                for j in range(self.J):
                    Xj=self.X[:,j].copy()  
                    f_loglik=self.F_1_SAEM(Xj,A_sample_long) 
                    def update_f_old_1(x,f_old=f_old_1[j],f_new=f_loglik):
                        return (1-step)*f_old(x)+step*f_new(x)
                    f_old_1[j]=update_f_old_1
                    def f_j(beta_gamma):
                        return f_old_1[j](beta_gamma)+self.ftn_pen(beta_gamma)
                    #progress_bar = tqdm(total=max_opt_iter, desc=f"Optimizing Parameter {j}")
                    #def callback(xk):
                    #    progress_bar.update(1)
                    opt_result=minimize(f_j,self.B_hat[j, :].flatten(),bounds=list(zip(lb,ub)),method='SLSQP',options=options)
                    #progress_bar.close()
                    B_update[j,:]=opt_result.x
                err=np.linalg.norm(self.B_hat-np.concatenate((B_update[:,[0]],thres(B_update[:,1:],self.tau)),axis=1),'fro')**2
                self.B_hat=np.concatenate((B_update[:,[0]],thres(B_update[:,1:],self.tau)),axis=1)
                t+=1
                #progress_bar.update(1)
                print(f'SAEM Iteration {t}, Err {err:.5f}')
                iter_indicator=(abs(err)>self.tol and t<self.max_iter)     
            U_T = (1 << self.K) - len(counts_smooth)
            nu_dense = np.zeros(1 << self.K, dtype=float)

            if counts_smooth:
                idxs = np.fromiter(counts_smooth.keys(), dtype=np.int64)
                vals = np.fromiter((counts_smooth[k] for k in idxs), dtype=float)
                nu_dense[idxs] = vals

            if self.L_final > 0.0 and U_T > 0:
                nu_dense += (self.L_final / U_T)

            total = nu_dense.sum()
            if total > 0:
                nu_dense /= total
            else:
                nu_dense.fill(1.0 / (1 << self.K))

            self.p_hat = nu_dense.reshape(-1, 1)
            #progress_bar.close()
            return self.p_hat,self.B_hat,self.A_hat,t,loglik
    def init(self,ite):
        self.generate_latent_data()
        #Record=ges(self.A,score_func='local_score_BDeu')
        #pyd=GraphUtils.to_pydot(Record['G'])
        #pyd.write_png(f'Q_A_{self.N}_ges_BDeu_{self.distribution}_{self.DAG_type}_{ite}.png')
        self.generate_data()
        nu_in=np.zeros(2**self.K) 
        A_src=binary(np.arange(2**self.K),self.K)
        gamma_in=np.zeros(self.J)
        G_est=np.vstack((np.eye(self.K),np.eye(self.K),np.eye(self.K))) 
        if self.distribution=='Poisson':
            U,S,Vt=np.linalg.svd(np.log(self.X+1),full_matrices=False)
            m=max(self.K+1,np.sum(np.diag(S)>1.01*np.sqrt(self.N)))
            X_top_m=U[:,:m]@np.diag(S[:m])@Vt[:m,:]
            X_top_m=np.maximum(X_top_m,self.epsilon)
            X_inv=np.exp(X_top_m)-1
        elif self.distribution=='Lognormal':
            X_inv=np.log(self.X)
        elif self.distribution=='Bernoulli':
            U,S,Vt=np.linalg.svd(self.X,full_matrices=False)
            m=max(self.K+1,np.sum(np.diag(S)>1.01*np.sqrt(self.N)))
            X_top_m=U[:,:m]@np.diag(S[:m])@Vt[:m,:]
            X_top_m=np.maximum(X_top_m,self.epsilon)
            X_top_m=np.minimum(X_top_m,1-self.epsilon)   
            X_inv=logit(X_top_m)
        X_inv_adj=X_inv-np.mean(X_inv,axis=0)  
        _,_,V_adj=np.linalg.svd(X_inv_adj,full_matrices=False)
        V_adj=V_adj.T
        rotator=Rotator(method='varimax')
        R_V=rotator.fit_transform(V_adj[:,:self.K])
        threshold=1/(2.5*np.sqrt(self.J))
        B_est=thres(R_V[:,:self.K],threshold)
        mean_per_column=B_est.mean(axis=0)
        sign_flip=2*(mean_per_column>0)-1  
        B_est=B_est*sign_flip#*np.sqrt(N)
        G_est=(B_est!=0).astype(int)
        col_perm=np.zeros(self.K,dtype=int)
        remaining_cols=list(range(self.K))
        for k in range(self.K):
                tmp=np.argmax(np.sum(G_est[[k,self.K+k,2*self.K+k],:][:,remaining_cols],axis=0))
                col_perm[k]=remaining_cols[tmp]
                del remaining_cols[tmp]
        B_est=B_est[:,col_perm]
        G_est=(B_est!=0).astype(float)
        A_est=X_inv_adj@B_est@inv(B_est.T@B_est)
        A_est=(A_est>0).astype(float)
        A_centered=A_est-np.ones((self.N,1))*np.mean(A_est, axis=0)
        B_re_est=(inv(A_centered.T@A_centered)@A_centered.T@X_inv_adj).T
        B_re_est=thres(B_re_est*G_est,0)
        if self.distribution=='Poisson':
            b=np.mean(X_inv,axis=0)-B_re_est@np.mean(A_est,axis=0)
        elif self.distribution=='Lognormal':
            b=np.mean(X_inv_adj,axis=0)-B_re_est@np.mean(A_est,axis=0)
        elif self.distribution=='Bernoulli':
            b=np.mean(X_inv_adj,axis=0)-B_re_est@np.mean(A_est,axis=0)
        B_ini=np.column_stack((b,B_re_est))
        rows_as_tuples=[tuple(row) for row in A_est]
        row_counts_dict={row: rows_as_tuples.count(row) for row in set(rows_as_tuples)}
        nu_in= (np.array([row_counts_dict.get(tuple(row),0) for row in A_src]))/self.N
        A_long=np.hstack((np.ones((self.N,1)),A_est))
        Tr=X_inv-A_long@B_ini.T
        for j in range(self.J):
                gamma_in[j]=np.sum(Tr[:,j]**2)/self.N
        return nu_in.reshape(-1,1),B_ini,gamma_in.reshape(-1,1),A_est    

    def estimate(self,ite,pyd_true):
        if self.distribution=='Poisson':
            if self.algorithm=='PSAEM':
                self.p_hat,self.B_hat,self.A_hat,t,loglik=self.PSAEM(ite)
            elif self.algorithm=='PEM':
                self.p_hat,self.B_hat,loglik,t=self.PEM(ite)
        elif self.distribution=='Lognormal':
            if self.algorithm=='PSAEM':
                self.p_hat,self.B_hat,self.gamma_hat,self.A_hat,t,loglik=self.PSAEM(ite)
            elif self.algorithm=='PEM':
                self.p_hat,self.B_hat,self.gamma_hat,loglik,t=self.PEM(ite)
        elif self.distribution=='Bernoulli':
            if self.algorithm=='PSAEM':
                self.p_hat,self.B_hat,self.A_hat,t,loglik=self.PSAEM(ite)
            elif self.algorithm=='PEM':
                self.p_hat,self.B_hat,loglik,t=self.PEM(ite)
        sam=np.zeros((self.Q_N,self.K))
        qwe=self.p_hat.flatten()
        counts=np.random.multinomial(self.Q_N, qwe)
        n=0
        A_src=binary(np.arange(2**self.K),self.K)
        for a in range(2**self.K):
            sam[n:n+(counts[a]).astype(int),:self.K]=np.tile(A_src[a,:self.K],((counts[a]).astype(int),1))
            n+=counts[a].astype(int)
        Record=ges(sam,score_func='local_score_BDeu')
        cpdag_ZZ_est_from_SAEM = Record['G'].graph
        ZX_est_SAEM = (self.B_hat[:, 1:] != 0).astype(int).T
        A_est_SAEM = stitch_full_A(cpdag_ZZ_est_from_SAEM, ZX_est_SAEM, n_x=self.J)
        
        sam=np.zeros((self.Q_N*2,self.K))
        counts=np.random.multinomial(self.Q_N*2, qwe)
        n=0
        A_src=binary(np.arange(2**self.K),self.K)
        for a in range(2**self.K):
            sam[n:n+(counts[a]).astype(int),:self.K]=np.tile(A_src[a,:self.K],((counts[a]).astype(int),1))
            n+=counts[a].astype(int)
        Record=ges(sam,score_func='local_score_BDeu')
        cpdag_ZZ_est_from_SAEM2 = Record['G'].graph
        ZX_est_SAEM = (self.B_hat[:, 1:] != 0).astype(int).T
        A_est_SAEM2 = stitch_full_A(cpdag_ZZ_est_from_SAEM2, ZX_est_SAEM, n_x=self.J)
        
        sam=np.zeros((self.Q_N*3,self.K))
        counts=np.random.multinomial(self.Q_N*3, qwe)
        n=0
        A_src=binary(np.arange(2**self.K),self.K)
        for a in range(2**self.K):
            sam[n:n+(counts[a]).astype(int),:self.K]=np.tile(A_src[a,:self.K],((counts[a]).astype(int),1))
            n+=counts[a].astype(int)
        Record=ges(sam,score_func='local_score_BDeu')
        cpdag_ZZ_est_from_SAEM3 = Record['G'].graph
        ZX_est_SAEM = (self.B_hat[:, 1:] != 0).astype(int).T
        A_est_SAEM3 = stitch_full_A(cpdag_ZZ_est_from_SAEM3, ZX_est_SAEM, n_x=self.J)
        A_truth = stitch_full_A(pyd_true.graph, (self.Q).T, n_x=self.J)
        shd_val  = shd_cpdag(A_truth, A_est_SAEM)
        shd2_val = shd_cpdag(A_truth, A_est_SAEM2)
        shd3_val  = shd_cpdag(A_truth, A_est_SAEM3)
        return shd_val,shd2_val,shd3_val


def stitch_full_A(ZZ, ZX, n_x):
    n_z = ZZ.shape[0]
    assert ZX.shape == (n_z, n_x)
    A = np.block([[ZZ, ZX],
                  [np.zeros((n_x, n_z), dtype=int), np.zeros((n_x, n_x), dtype=int)]])
    np.fill_diagonal(A, 0)
    return A
def shd_cpdag(M1, M2):
    """
    计算两个 CPDAG 矩阵(因果学习语义)之间的一步编辑距离：
      - 无 <-> 有 (含有向/无向) 计 1
      - 无向 <-> 有向 计 1
      - i->j <-> j->i 计 1
      - 完全一致计 0
    """
    n = M1.shape[0]
    def code(M, i, j):
        a, b = M[i, j], M[j, i]
        if a == 0 and b == 0:
            return 0           # none
        if a == -1 and b == -1:
            return 2           # undirected
        if b == 1 and a == -1:
            return 1           # i->j
        if a == 1 and b == -1:
            return -1          # j->i  (与上相反方向)
        # 其它异常情形按“有边但未知”处理成无向
        return 2
    shd = 0
    for i in range(n):
        for j in range(i + 1, n):
            c1 = code(M1, i, j)
            c2 = code(M2, i, j)
            if c1 == c2:
                continue
            # 不同即加 1
            shd += 1
    return shd



class ParallelDAGEstimator(DAG_Estimator):
    def __init__(self,N,J,K,Q_type,DAG_type,distribution,algorithm,upb,lob,lob2,upb2,lob3,upb3,lob4,upb4,lob5,upb5,tau,pen,max_iter,tol,C,epsilon,Q_N,kappa,L_final):
        super().__init__(N,J,K,Q_type,DAG_type,distribution,algorithm,upb,lob,lob2,upb2,lob3,upb3,lob4,upb4,lob5,upb5,tau,pen,max_iter,tol,C,epsilon,Q_N,kappa,L_final)

    def run_and_log(self,i,pyd_true,output_path):
        try:
            res=self.estimate(i,pyd_true)
        except Exception as e:
            print(f"[Warning] Iteration {i} failed: {e}")
            res=("ERROR",)*12

        res_line=[str(i)]+[f"{x:.6f}" if isinstance(x, float) else str(x) for x in res]
        line=",".join(res_line)+"\n"
        with FileLock(output_path+".lock"):
            with open(output_path,"a") as f:
                f.write(line)

    def parallel_estimate_streaming(self,num_iterations,pyd_true,output_path=None):
        results_dir = "results"          # or "outputs", or "/scratch/$USER/results"
        os.makedirs(results_dir, exist_ok=True)   # make sure it exists

    # 2. build filename inside that directory
        if output_path is None:
            filename = f"results_{self.N}_{self.distribution}_{self.DAG_type}_{self.Q_type}.txt"
            output_path = os.path.join(results_dir, filename)
        if not os.path.exists(output_path):
            with open(output_path, "w") as f:
                header=[
                    "iter",
                    "shd_val","shd2_val","shd3_val"
                ]
                f.write(",".join(header)+"\n")
                
        Parallel(n_jobs=5)(
            delayed(self.run_and_log)(i,pyd_true,output_path)
            for i in num_iterations
        )

dag_model=Generate_Data(N=4000,J=45,K=15,Q_type='2',DAG_type='DiverseTree',distribution='Lognormal',upb=0.65,lob=0.6,upb2=0.4,lob2=0.35,upb3=0.82,lob3=0.77,upb4=0.25,lob4=0.2,upb5=0.7,lob5=0.65)
A1=dag_model.generate_latent_data()
Record=ges(A1,score_func='local_score_BDeu')
true_DiverseTree=Record['G']
#pyd_true_DiverseTree=GraphUtils.to_pydot(Record['G'])
#pyd_true_DiverseTree.write_png(f'DiverseTree.png')

dag_model=Generate_Data(N=4000,J=30,K=10,Q_type='2',DAG_type='Tree',distribution='Lognormal',upb=0.65,lob=0.6,upb2=0.4,lob2=0.35,upb3=0.82,lob3=0.77,upb4=0.25,lob4=0.2,upb5=0.7,lob5=0.65)
A1=dag_model.generate_latent_data()
Record=ges(A1,score_func='local_score_BDeu')
true_Tree=Record['G']
#pyd_true_Tree=GraphUtils.to_pydot(Record['G'])
#pyd_true_Tree.write_png(f'Tree.png')

dag_model=Generate_Data(N=4000,J=30,K=10,Q_type='2',DAG_type='Markov',distribution='Lognormal',upb=0.65,lob=0.6,upb2=0.4,lob2=0.35,upb3=0.82,lob3=0.77,upb4=0.25,lob4=0.2,upb5=0.7,lob5=0.65)
A1=dag_model.generate_latent_data()
Record=ges(A1,score_func='local_score_BDeu')
true_Markov=Record['G']
#pyd_true_Markov=GraphUtils.to_pydot(Record['G'])
#pyd_true_Markov.write_png(f'Markov.png')

dag_model=Generate_Data(N=4000,J=24,K=8,Q_type='2',DAG_type='Model-8',distribution='Lognormal',upb=0.65,lob=0.6,upb2=0.4,lob2=0.35,upb3=0.82,lob3=0.77,upb4=0.25,lob4=0.2,upb5=0.7,lob5=0.65)
A1=dag_model.generate_latent_data()
Record=ges(A1,score_func='local_score_BDeu')
true_Model8=Record['G']
#pyd_true_Model8=GraphUtils.to_pydot(Record['G'])
#pyd_true_Model8.write_png(f'Model-8.png')

dag_model=Generate_Data(N=4000,J=21,K=7,Q_type='2',DAG_type='Model-7',distribution='Lognormal',upb=0.65,lob=0.6,upb2=0.4,lob2=0.35,upb3=0.82,lob3=0.77,upb4=0.25,lob4=0.2,upb5=0.7,lob5=0.65)
A1=dag_model.generate_latent_data()
Record=ges(A1,score_func='local_score_BDeu')
true_Model7=Record['G']
#pyd_true_Model7=GraphUtils.to_pydot(Record['G'])
#pyd_true_Model7.write_png(f'Model-7.png')

dag_model=Generate_Data(N=4000,J=39,K=13,Q_type='2',DAG_type='Model-13',distribution='Lognormal',upb=0.65,lob=0.6,upb2=0.4,lob2=0.35,upb3=0.82,lob3=0.77,upb4=0.25,lob4=0.2,upb5=0.7,lob5=0.65)
A1=dag_model.generate_latent_data()
Record=ges(A1,score_func='local_score_BDeu')
true_Model13=Record['G']
#pyd_true_Model13=GraphUtils.to_pydot(Record['G'])
#pyd_true_Model13.write_png(f'Model-13.png')

N,K=args.n,args.k
J=3*K 
Q_type=2
DAG_type=args.dag_type
distribution=args.distribution
algorithm='PSAEM'
upb,lob,upb2,lob2,upb3,lob3,upb4,lob4,upb5,lob5=0.65,0.6,0.4,0.35,0.82,0.77,0.25,0.2,0.7,0.65
lambda_vec=[N**(1/8),N**(2/8),N**(3/8)]
const=0.9
tau_vec=2*N**(np.array([1/8,2/8,3/8])*const-1/2)
#if distribution=='Lognormal':
#    tau=tau_vec[2]
#    pen=lambda_vec[2]
#elif distribution=='Poisson':
#    tau=tau_vec[2]
#    pen=lambda_vec[1]
#else:
#    tau=tau_vec[1]
#    pen=lambda_vec[1]
tau=tau_vec[1]
pen=lambda_vec[1]
max_iter=20
if distribution=='Bernoulli':
    tol=0.5
else:
    tol=0.05
C=1
kappa=1e-200
L_final=0
if distribution=='Poisson':
    epsilon=1e-50
else:
    epsilon=1e-5
Q_N=N
if DAG_type=="DiverseTree":
    truth=true_DiverseTree
elif DAG_type=="Model-8":
    truth=true_Model8
elif DAG_type=="Model-7":
    truth=true_Model7
elif DAG_type=="Model-13":
    truth=true_Model13
elif DAG_type=="Tree":
    truth=true_Tree
elif DAG_type=="Markov":
    truth=true_Markov
else:
    raise ValueError("Unknown DAG_type")
dag_estimator=ParallelDAGEstimator(N,J,K,Q_type,DAG_type,distribution,algorithm,upb,lob,lob2,upb2,lob3,upb3,lob4,upb4,lob5,upb5,tau,pen,max_iter,tol,C,epsilon,Q_N,kappa,L_final)
dag_estimator.parallel_estimate_streaming(range(args.start,args.end),truth)
