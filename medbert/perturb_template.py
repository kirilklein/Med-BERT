"""
Created on Wed Nov 28 12:57:40 2018
@author: ginnyzhu
Last reviewed and updated Lrasmy Feb 21 2020
"""
from __future__ import print_function, division
from io import open
import string
import re
import random

import os
import argparse
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle
    
#import self-defined modules
#models, utils, and Dataloader
#sys.path.insert() only for jupyter notebook imports
import sys
sys.path.insert(0, '../ehr_pytorch_demo')
import models as model 
from EHRDataloader import EHRdataFromPickles,EHRdataFromLoadedPickles, EHRdataloader 
import utils as ut #:)))) 
from EHREmb import EHREmbeddings

#silly ones
from termcolor import colored
from tqdm import tqdm
# check GPU availability
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
#args, slightly modified from main.py file to be more compatible with jupyter notebook 
#all args provide default values, so you can run the whole notebook without changing/providing any args
#args ordered by dataloader, model, and training sections
#Result_DIR='/bgfs/lwang/StudentData/PTSD/SUD/'
#Result_DIR='/bgfs/lwang/StudentData/UPMC_AD_RESULT/DL/'
Result_DIR='/bgfs/lwang/StudentData/PTSD/SUD/'
#Result_DIR='/bgfs/lwang/StudentData/HF/Result/'
#Filename="EHRFormat.combined.train"
#Filename="EHRFormat_withoutlab.combined.all"
#Filename="EHRFormat_onlydx.combined.all"
#Filename="EHRFormat.combined.train"
#Result_DIR='/bgfs/lwang/StudentData/UPMC_AD_RESULT/'
Filename="EHRFormat_demo.combined.all"
def options():
    parser = argparse.ArgumentParser(description='Predictive Analytics on EHR with Pytorch')
    
    #EHRdataloader 
    parser.add_argument('-root_dir', type = str, default = Result_DIR , 
                        help='the path to the folders with pickled file(s)')
    parser.add_argument('-file', type = str, default = Filename , 
                        help='the name of pickled files')
    parser.add_argument('-test_ratio', type = float, default = 0.1, 
                        help='test data size [default: 0.2]')
    parser.add_argument('-valid_ratio', type = float, default = 0.1, 
                        help='validation data size [default: 0.1]')
    
    #EHRmodel
    parser.add_argument('-which_model', type = str, default = 'DRNN', 
                        help='choose from {"RNN","DRNN","QRNN","LR"}') 
    parser.add_argument('-cell_type', type = str, default = 'GRU', 
                        help='For RNN based models, choose from {"RNN", "GRU", "LSTM"}')
    parser.add_argument('-input_size', type = list, default =[15817], 
                        help='''input dimension(s), decide which embedding types to use. 
                        If len of 1, then  1 embedding; 
                        len of 3, embedding medical, diagnosis and others separately (3 embeddings) 
                        [default:[15817]]''') ###multiple embeddings not effective in this release
    parser.add_argument('-embed_dim', type=int, default=128, 
                        help='number of embedding dimension [default: 128]')
    parser.add_argument('-hidden_size', type=int, default=128, 
                        help='size of hidden layers [default: 128]')
    parser.add_argument('-dropout_r', type=float, default=0.1, 
                        help='the probability for dropout[default: 0.1]')
    parser.add_argument('-n_layers', type=int, default=3, 
                        help='''number of Layers, 
                        for Dilated RNNs, dilations will increase exponentialy with mumber of layers [default: 1]''')
    parser.add_argument('-bii', type=bool, default=False, 
                        help='indicator of whether Bi-directin is activated. [default: False]')
    parser.add_argument('-time', type=bool, default=False, 
                        help='indicator of whether time is incorporated into embedding. [default: False]')
    parser.add_argument('-preTrainEmb', type= str, default='', 
                        help='path to pretrained embeddings file. [default:'']')
    parser.add_argument("-output_dir",type=str, default= '../models/', 
                        help="The output directory where the best model will be saved and logs written [default: we will create'../models/'] ")
    
    # training 
    parser.add_argument('-lr', type=float, default=10**-4, 
                        help='learning rate [default: 0.0001]')
    parser.add_argument('-L2', type=float, default=10**-4, 
                        help='L2 regularization [default: 0.0001]')
    parser.add_argument('-epochs', type=int, default= 100, 
                        help='number of epochs for training [default: 100]')
    parser.add_argument('-patience', type=int, default= 20, 
                        help='number of stagnant epochs to wait before terminating training [default: 20]')
    parser.add_argument('-batch_size', type=int, default=128, 
                        help='batch size for training, validation or test [default: 128]')
    parser.add_argument('-optimizer', type=str, default='adam', 
                        choices=  ['adam','adadelta','adagrad', 'adamax', 'asgd','rmsprop', 'rprop', 'sgd'], 
                        help='Select which optimizer to train [default: adam]. Upper/lower case does not matter') 
    #parser.add_argument('-cuda', type= bool, default=True, help='whether GPU is available [default:True]')
    args = parser.parse_args([])
    return args 
args = options()
##Update the args here if you dont want to use the default ones
##start an example
args.which_model = 'RETAIN'
args.cell_type = 'LSTM'
args.embed_dim = 128
args.hidden_size = 128
args.dropout_r = 0.2
args.n_layers = 8
args.input_size=[5000]
args.patience=3
##end
print(args)
####Step1. Data preparation
#By default, prevent sort (on visit length) before splitting, if splitting
#Gotta specify your split ratios here if intend to split on non-default split ratios
#First load your data

print(colored("\nLoading and preparing data...", 'green'))    
data = EHRdataFromPickles(root_dir = args.root_dir, 
                          file = args.file, 
                          sort= False,
                          test_ratio = args.test_ratio, 
                          valid_ratio = args.valid_ratio) 
#depending on different models, model parameters might have different choices.
#e.g. if you set bi = True for DRNN or QRNN, it will throw you warnings and implement correct bi =False instead
if args.which_model == 'RNN': 
    ehr_model = model.EHR_RNN(input_size= args.input_size, 
                              embed_dim=args.embed_dim, 
                              hidden_size= args.hidden_size,
                              n_layers= args.n_layers,
                              dropout_r=args.dropout_r,
                              cell_type=args.cell_type,
                              bii= args.bii,
                              time= args.time,
                              preTrainEmb= args.preTrainEmb,
                              bSDOH=True) 
    pack_pad = True
elif args.which_model == 'DRNN': 
    ehr_model = model.EHR_DRNN(input_size= args.input_size, 
                              embed_dim=args.embed_dim, 
                              hidden_size= args.hidden_size,
                              n_layers= args.n_layers,
                              dropout_r=args.dropout_r, #default =0 
                              cell_type=args.cell_type, #default ='DRNN'
                              bii= False,
                              time = args.time, 
                              preTrainEmb= args.preTrainEmb,
                              bSDOH=True)     
    pack_pad = False
elif args.which_model == 'QRNN': 
    ehr_model = model.EHR_QRNN(input_size= args.input_size, 
                              embed_dim=args.embed_dim, 
                              hidden_size= args.hidden_size,
                              n_layers= args.n_layers,
                              dropout_r=args.dropout_r, #default =0.1
                              cell_type= 'QRNN', #doesn't support normal cell types
                              bii= False, #QRNN doesn't support bi
                              time = args.time,
                              preTrainEmb= args.preTrainEmb,
                              bSDOH=True)  
    pack_pad = False
elif args.which_model == 'TLSTM': 
    ehr_model = model.EHR_TLSTM(input_size= args.input_size, 
                              embed_dim=args.embed_dim, 
                              hidden_size= args.hidden_size,
                              n_layers= args.n_layers,
                              dropout_r=args.dropout_r, #default =0.1
                              cell_type= 'TLSTM', #doesn't support normal cell types
                              bii= False, 
                              time = args.time, 
                              preTrainEmb= args.preTrainEmb,
                               bSDOH=True)  
    pack_pad = False
elif args.which_model == 'RETAIN': 
    ehr_model = model.RETAIN(input_size= args.input_size, 
                              embed_dim=args.embed_dim, 
                              hidden_size= args.hidden_size,
                              n_layers= args.n_layers,
                            bSDOH=True) 
    pack_pad = False
else: 
    ehr_model = model.EHR_LR_emb(input_size = args.input_size,
                                 embed_dim = args.embed_dim,
                                 preTrainEmb= args.preTrainEmb,
                                bSDOH=True)
    pack_pad = False


#make sure cuda is working
if use_cuda:
    ehr_model = ehr_model.cuda() 
#model optimizers to choose from. Upper/lower case dont matter
if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(ehr_model.parameters(), 
                           lr=args.lr, 
                           weight_decay=args.L2)
elif args.optimizer.lower() == 'adadelta':
    optimizer = optim.Adadelta(ehr_model.parameters(), 
                               lr=args.lr, 
                               weight_decay=args.L2)
elif args.optimizer.lower() == 'adagrad':
    optimizer = optim.Adagrad(ehr_model.parameters(), 
                              lr=args.lr, 
                              weight_decay=args.L2) 
elif args.optimizer.lower() == 'adamax':
    optimizer = optim.Adamax(ehr_model.parameters(), 
                             lr=args.lr, 
                             weight_decay=args.L2)
elif args.optimizer.lower() == 'asgd':
    optimizer = optim.ASGD(ehr_model.parameters(), 
                           lr=args.lr, 
                           weight_decay=args.L2)
elif args.optimizer.lower() == 'rmsprop':
    optimizer = optim.RMSprop(ehr_model.parameters(), 
                              lr=args.lr, 
                              weight_decay=args.L2)
elif args.optimizer.lower() == 'rprop':
    optimizer = optim.Rprop(ehr_model.parameters(), 
                            lr=args.lr)
elif args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(ehr_model.parameters(), 
                          lr=args.lr, 
                          weight_decay=args.L2)
else:
    raise NotImplementedError
#if you want to use previous trained models, use
best_model= torch.load(args.output_dir + 'dhf.trainEHRmodel.pth')
best_model.load_state_dict(torch.load(args.output_dir + 'dhf.trainEHRmodel.st'))
best_model.eval()
all_mbs = list(tqdm(EHRdataloader(data, batch_size = 1, packPadMode = pack_pad)))
#if you want to shuffle batches before using them, add this line 
#(options are achieved in utils by setting shuffle = True)
for param in best_model.parameters():
    param.requires_grad = False
best_model.train()
def Phi(embedded, age, gender, nSES ):
    global best_model
    #x=x.unsqueeze(0)
    out=best_model.forward4(embedded, age, gender, nSES)

    return out
def calculate_regularization(sampled_x, Phi, reduced_axes=None, device=None):
    """ Calculate the variance that is used for Interpreter

    Args:
        sample_x (list of torch.FloatTensor):
            A list of sampled input embeddings $x$, each $x$ is of shape ``[length, dimension]``. All the $x$s can have different length,
            but should have the same dimension. Sampled number should be higher to get a good estimation.
        Phi (function):
            The $Phi$ we studied. A function whose input is x (element in the first parameter) and returns a hidden state (of type 
            ``torch.FloatTensor``, of any shape)
        reduced_axes (list of ints, Optional):
            The axes that is variable in Phi (e.g., the sentence length axis). We will reduce these axes by mean along them.
    
    Returns:
        torch.FloatTensor: The regularization term calculated

    """
    global best_model
    sample_s = []
    for i, batch in enumerate(sampled_x): 
        #mb_t, lbt_t,seq_l, mtd=batch
        mb_t, lbt_t,seq_l, mtd, age, gender, nSES = batch
        out1=best_model.embed(mb_t)
        #out1=best_model.EmbedPatient_MB(mb_t,mtd)

        #out1=out1.squeeze()
        
        
    #sample_num = len(sampled_x)
    #sample_s = []
    #for n in range(sample_num):
    #    x = sampled_x[n]
    #    if device is not None:
    #        x = x.to(device)
        s = Phi(out1, age, gender, nSES)
        if reduced_axes is not None:
            for axis in reduced_axes:
                assert axis < len(s.shape)
                s = s.mean(dim=axis, keepdim=True)
        sample_s.append(s.tolist())
    sample_s = np.array(sample_s)
    return np.std(sample_s, axis=0)

class Interpreter(nn.Module):
    """ Interpreter for interpret one instance.

    It will minimize the loss in Eqn.(7):

        $L(sigma) = (||Phi(embed + epsilon) - Phi(embed)||_2^2) // (regularization^2) - rate * log(sigma)$

    In our implementation, we use reparameterization trick to represent epsilon ~ N(0, sigma^2 I), i.e. epsilon = scale * ratio * noise.
    Where noise ~ N(0, 1), scale is a hyper-parameter that controls the maximum value of sigma^2, and ratio in (0, 1) is the learnable parameter.

    """
    def __init__(self, x,age, gender, nSES, mask, Phi, scale=0.5, rate=0.1, regularization=None, words=None):
        """ Initialize an interpreter class.

        Args:
            x (torch.FloatTensor): Of shape ``[length, dimension]``.
                The $x$ we studied. i.e. The input word embeddings.
            Phi (function):
                The $Phi$ we studied. A function whose input is x (the first parameter) and returns a hidden state (of type ``torch.FloatTensor``, of any shape)
            scale (float):
                The maximum size of sigma. A hyper-parameter in reparameterization trick. The recommended value is 10 * Std[word_embedding_weight], 
                where word_embedding_weight is the word embedding weight in the model interpreted. Larger scale will give more salient result, Default: 0.5.
            rate (float):
                A hyper-parameter that balance the MLE Loss and Maximum Entropy Loss. Larger rate will result in larger information loss. Default: 0.1.
            regularization (Torch.FloatTensor or np.ndarray):
                The regularization term, should be of the same shape as (or broadcastable to) the output of Phi. If None is given, method will use the output to 
                regularize itself. Default: None.
            words (List[Str]):
                The input sentence, used for visualizing. If None is given, method will not show the words.

        """
        super(Interpreter, self).__init__()
        self.d1 = x.size(0)
        self.d2 = x.size(1)
        self.d3 = x.size(2)
        self.d4 = x.size(3)
        self.ratio = nn.Parameter(torch.randn(self.d2, self.d3), requires_grad=True)

        self.scale = scale
        self.rate = rate
        self.mask = mask
        self.x = x
        self.age=age
        self.gender=gender
        self.nSES=nSES
        self.Phi = Phi

        self.regular = regularization
        if self.regular is not None:
            self.regular = nn.Parameter(torch.tensor(self.regular).to(x), requires_grad=False)
        self.words = words
        if self.words is not None:
            assert self.s == len(words), 'the length of x should be of the same with the lengh of words'

    def forward(self):
        """ Calculate loss:
        
            $L(sigma) = (||Phi(embed + epsilon) - Phi(embed)||_2^2) // (regularization^2) - rate * log(sigma)$

        Returns:
            torch.FloatTensor: a scalar, the target loss.

        """
        ratios = torch.sigmoid(self.ratio)  # S * 1


        
        x = self.x + 0.    # S * D
        #print(x.shape)
        ratios=torch.unsqueeze(ratios, 0)
        ratios=torch.unsqueeze(ratios, 3)
        ratios = ratios.expand(x.shape)
        mask=self.mask
       
        mask=torch.unsqueeze(mask, 3)
        mask = mask.expand(x.shape)
        
        #print(ratios.shape)
        x_tilde = x + mask*ratios * torch.randn(self.d1, self.d2, self.d3, self.d4).to(x.device) * self.scale  # S * D
        age=self.age
        gender=self.gender
        nSES=self.nSES
        s = self.Phi(x, self.age, self.gender, self.nSES)  # D or S * D
        s_tilde = self.Phi(x_tilde, self.age, self.gender, self.nSES)
        loss = (s_tilde - s) ** 2
        if self.regular is not None:
            loss = torch.mean(loss / self.regular ** 2)
        else:
            loss = torch.mean(loss) / torch.mean(s ** 2)

        return loss - torch.mean(torch.log(ratios)) * self.rate

    def optimize(self, iteration=5000, lr=0.01, show_progress=False):
        """ Optimize the loss function

        Args:
            iteration (int): Total optimizing iteration
            lr (float): Learning rate
            show_progress (bool): Whether to show the learn progress

        """
        minLoss = None
        state_dict = None
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        func = (lambda x: x) if not show_progress else tqdm
        for _ in func(range(iteration)):
            optimizer.zero_grad()
            loss = self()
            
            loss.backward()
            optimizer.step()
            if minLoss is None or minLoss > loss:
                state_dict = {k:self.state_dict()[k] + 0. for k in self.state_dict().keys()}
                minLoss = loss
        self.eval()
        self.load_state_dict(state_dict)

    def get_sigma(self):
        """ Calculate and return the sigma

        Returns:
            np.ndarray: of shape ``[seqLen]``, the ``sigma``.

        """
        ratios = torch.sigmoid(self.ratio)  # S * 1
        mask=self.mask
        mask=mask.squeeze()
        ratios=mask*ratios
        return ratios.detach().cpu().numpy() * self.scale
    
    def visualize(self):
        """ Visualize the information loss of every word.
        """
        sigma_ = self.get_sigma()
        _, ax = plt.subplots()
        ax.imshow([sigma_], cmap='GnBu_r')
        ax.set_xticks(range(self.s))
        ax.set_xticklabels(self.words)
        ax.set_yticks([0])
        ax.set_yticklabels([''])
        plt.tight_layout()
        plt.show()
def iter_batch2(iterable, samplesize):
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    for _ in range(samplesize):
        results.append(iterator.__next__())
    #random.shuffle(results)  
    return results
loader2 = iter_batch2(all_mbs, len(all_mbs))
#print(len(dataset))
#input a 3 d matrix and output a mask matrix
def my_mask(source):

    for i in range(len(source[1])):
        for j in range(len(source[2])):
            if source[0][i][j] > 0:
                source[0][i][j] = 1
            else:
                 source[0][i][j] = 0
    return source

def calculate_regularization(sampled_x, Phi, reduced_axes=None, device=None):
    """ Calculate the variance that is used for Interpreter

    Args:
        sample_x (list of torch.FloatTensor):
            A list of sampled input embeddings $x$, each $x$ is of shape ``[length, dimension]``. All the $x$s can have different length,
            but should have the same dimension. Sampled number should be higher to get a good estimation.
        Phi (function):
            The $Phi$ we studied. A function whose input is x (element in the first parameter) and returns a hidden state (of type 
            ``torch.FloatTensor``, of any shape)
        reduced_axes (list of ints, Optional):
            The axes that is variable in Phi (e.g., the sentence length axis). We will reduce these axes by mean along them.
    
    Returns:
        torch.FloatTensor: The regularization term calculated

    """
    global best_model

    sample_s = []
    for i, batch in enumerate(sampled_x): 
        mb_t, lbt_t,seq_l, mtd, age, gender, nSES = batch
        out1=best_model.embed(mb_t)

        s = Phi(out1, age, gender, nSES)
        if reduced_axes is not None:
            for axis in reduced_axes:
                assert axis < len(s.shape)
                s = s.mean(dim=axis, keepdim=True)
        sample_s.append(s.tolist())
    sample_s = np.array(sample_s)
    return np.std(sample_s, axis=0)


# calculate the regularization term.
device='cuda:0'
#regularization = calculate_regularization(loader2, Phi, device=device)
#print(regularization)
import time
import numpy
t0= time.time()

istart=0
batchsize=1000
regularization=[[0.4700104]]
import pandas as pd
#resultfilename=Result_DIR+str(istart)+'_v2.result'
MyList=[]
#xdata=pd.read_csv(resultfilename,delimiter=' ', header=None, usecols=[3], names=["indexA"])
#lastrow=xdata.iloc[-1]
#for j in range(lastrow['indexA']+1, istart+1800 ):
for j in range(istart, istart+batchsize ):
    MyList+=[j,]

myrange=MyList
resultfilename=Result_DIR+str(istart)+'.resultPTSD_SUDSDOH'#resultfilename+'1'
def my_mask(source):

    for i in range(source.shape[1]):
        for j in range(source.shape[2]):
            if source[0][i][j] > 0:
                source[0][i][j] = 1
            else:
                 source[0][i][j] = 0
    return source


device='cuda:0'
resultlist=[]
index=0
for i, batch in enumerate(loader2):  
    if index in myrange:
        mb_t, lbt_t,seq_l, mtd, age, gender, nSES = batch
        if index % 100==0:
            print("index="+str(index))
            if len(resultlist)>0:
                numpy.savetxt(resultfilename,  resultlist, fmt='%d %f %d %d', delimiter=",")
        out1=best_model.embed(mb_t)
        print(out1.shape)
        mb_t_numpy=mb_t.cpu().detach().numpy()
        lbt_t_numpy=lbt_t.cpu().detach().numpy()
        mb_t_numpy_copy=mb_t_numpy.copy()
        mask1=my_mask(mb_t_numpy_copy)
        mask=torch.from_numpy(mask1)
        mask=mask.to(device)
        interpreter = Interpreter(
        x=out1,age=age, gender=gender, nSES=nSES, mask=mask, Phi=Phi, regularization=regularization, scale=10 * 0.1, words=None)
        interpreter.to(device)
        interpreter.optimize(iteration=10000, lr=0.01, show_progress=False)
        mylist=interpreter.get_sigma()
        for j in range(mask1.shape[1]):
            for k in range(mask1.shape[2]):
                if mask1[0][j][k] == 1:
                    resultlist.append([mb_t_numpy[0][j][k], mylist[j][k], lbt_t_numpy[0], index ] )
    index=index+1
numpy.savetxt(resultfilename,  resultlist, fmt='%d %f %d %d', delimiter=",")
t1 = time.time() - t0
print("Time elapsed: ", t1)