import numpy as np
import torch
from torch import nn
import os
import random
import itertools
from torch.utils.data import Dataset,DataLoader
from torch.nn.functional import log_softmax,softmax
from sklearn.model_selection import train_test_split,GridSearchCV
from numpy.random import default_rng
import pickle
from scipy.io import loadmat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,MaxAbsScaler

class EncoderNet(nn.Module):
    def __init__(self,N,middleDim):
        super(EncoderNet,self).__init__()
        self.N = N
        self.middleDim = middleDim
        self.lin1 = nn.Linear(20,self.middleDim) # nn.Identity() self.N
        self.nrm1 =nn.BatchNorm1d(self.middleDim,eps=5e-5)
        self.lin2 = nn.Linear(self.middleDim,self.middleDim)
        self.nrm2= nn.BatchNorm1d(self.middleDim,eps=5e-5)
        self.lin3 = nn.Linear(self.middleDim,32) #20
        self.act1 = nn.LeakyReLU(0.7) #nn.ReLU()#nn.GELU()
    def forward(self,x):
        out1 = self.act1(self.nrm1(self.lin1(x)))
        out2 = self.act1(self.nrm2(self.lin2(out1)))
        out3 = self.lin3(out2)
        return out3

class DecoderNet(nn.Module):
    def __init__(self,middleDim,N=78,valCats = None):
        super(DecoderNet,self).__init__()
        self.N = N+2
        self.middleDim = middleDim
        self.lin1 = nn.Linear(24,self.middleDim)# 15
        self.nrm1 = nn.BatchNorm1d(self.middleDim,eps=5e-5)
        self.lin2 = nn.Linear(self.middleDim,self.middleDim)
        self.nrm2 = nn.BatchNorm1d(self.middleDim,eps=5e-5)
        self.lin3 = nn.Linear(self.middleDim,N)#109 20 30 # N
        self.act1 = nn.LeakyReLU(0.7) #nn.ReLU()#nn.GELU()
        self.act2= nn.Softplus(beta=1,threshold=20)
    def forward(self,x):
        out1 = self.act1(self.nrm1(self.lin1(x)))
        out2 = self.act1(self.nrm2(self.lin2(out1)))
        out3 = self.act2(self.lin3(out2)) #self.act1(
        return out3

class ClfModel(nn.Module):
    def __init__(self,inputDim,numClasses=12,drp_out_param=0.7):
      super(ClfModel,self).__init__()
      self.lin_stack = nn.Sequential(
          nn.Linear(inputDim,40),
          nn.ReLU(),
          nn.BatchNorm1d(40),
          nn.Dropout(drp_out_param),#originally 0.4
          nn.Linear(40,30),
          nn.ReLU(),
          nn.BatchNorm1d(30),
          nn.Dropout(drp_out_param), # originally 0.4
          nn.Linear(30,20)
      )
      # self.lin_stack = nn.Sequential(
      #     nn.Linear(inputDim,40),
      #     nn.LeakyReLU(0.7),
      #     nn.LayerNorm(40),
      #     nn.Dropout(drp_out_param),#originally 0.4
      #     nn.Linear(40,30),
      #     nn.LeakyReLU(0.7),
      #     nn.LayerNorm(30),
      #     nn.Dropout(drp_out_param), # originally 0.4
      #     nn.Linear(30,20)
      # )
      self.lin = nn.Linear(20,numClasses)
      self.act  = nn.Softmax(dim=1)
      self.clf_loss =nn.CrossEntropyLoss()
    def produce_latent(self,x):
      return self.lin_stack(x)
    def forward(self,x,val):
      out = self.lin_stack(x)
      out2  = self.lin(out)
      out3  = self.act(out2)
      loss = self.clf_loss(out3,val)
      return loss


class Encoder(nn.Module):
    def __init__(self,encoder_net,device,content_dim):
        super(Encoder,self).__init__()
        self.encoder = encoder_net
        self.device = device
        self.content_dim = content_dim + 2
    def reparameterization(self,mu,log_var):
        z = mu + torch.exp(0.5*log_var)*torch.randn(log_var.shape).to(self.device)
        return z
    def encode(self,x):
        out = self.encoder(x)
        # split the encoder output into 3 parts
        content_part = out[:,:out.shape[1]//2]
        mu,log_var = torch.chunk(out[:,out.shape[1]//2:],2,dim=1)
        return content_part,mu,log_var

    def sample(self,x=None,mu=None,log_var=None,content_part=None):
        # encode the input
        if x != None:
            content_part,mu,log_var = self.encode(x)
        # apply the reparameterization trick
        z = self.reparameterization(mu,log_var)
        return torch.cat((content_part,z),axis=1)

    def log_prob(self,x=None,mu=None,log_var=None,content_part=None):
        if x != None:
            content_part,mu,log_var = self.encode(x)
        if content_part == None and x == None:
            content_part = torch.zeros(mu.shape)
        z = self.sample(mu=mu,log_var=log_var,content_part=content_part)
        var_term = torch.stack([torch.diag(torch.exp(0.5*log_var[i,:])) for i in range(log_var.shape[0])])
        m = torch.distributions.multivariate_normal.MultivariateNormal(mu,var_term)
        return m.log_prob(z[:,self.content_dim:])

    def forward(self,x,type='log_prob'):
         if type == 'log_prob':
             return self.log_prob(x)
         else:
            return self.sample(x)

class Decoder(nn.Module):
    def __init__(self,decoder_net,N):#,numCountVals
        # Note: We are assuming a categorical distribution
        super(Decoder,self).__init__()
        self.decoder = decoder_net
        # self.numCountVals = numCountVals
        self.N = N
    def decode(self,z):
        out = self.decoder(z)
        return out
    def forward(self,z):
        return self.decode(z)

class Prior(nn.Module):
    def __init__(self,numCountVals,device):
        super(Prior,self).__init__()
        self.numCountVals = numCountVals
        self.device = device
        self.distr = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(self.numCountVals).to(self.device), torch.eye(self.numCountVals).to(self.device))
    def sample(self,batch_size):
        return torch.randn((batch_size,self.numCountVals))
    def log_prob(self,x):
        return self.distr.log_prob(x)

class BlockSwapVAE_N(nn.Module):
    def __init__(self,encoder,decoder,N,beta,alpha,device,embed_file,embed_flag = True,inv_cov = None,numClasses=None,numUnits=78,FreezeFlag = True):#,numCountVals
        super(BlockSwapVAE_N,self).__init__()
        self.device = device
        self.encoder = Encoder(encoder,self.device,N)#,numCountVals
        self.decoder = Decoder(decoder,N=numUnits)#,numCountVals
        self.N = N+2
        self.inv_cov = inv_cov
        self.embed_flag = embed_flag
        # self.numCountVals = numCountVals
        self.prior = Prior(self.N,self.device)#Prior(2*self.N,self.device)
        self.rec_loss = nn.HuberLoss(delta=1.35)
        self.align_loss = nn.HuberLoss(delta=1.35)
        self.beta = beta
        self.alpha = alpha
        self.embed = ClfModel(numUnits,numClasses=numClasses)#Model2(latentDim=20,inputDim=20,inv_cov = self.inv_cov) #20 27
        self.embed.load_state_dict(torch.load(embed_file,map_location=torch.device('cpu')))
        if FreezeFlag == True:
          for name,params in self.embed.named_parameters():
            params.requires_grad = False
          self.embed.eval()
    def randTrfm(self,x):
      noise = torch.normal(0,0.2,size = x.shape)
      new_x = torch.clone(x)+ noise.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
      new_x[torch.FloatTensor(x.shape).uniform_() > 0.6] = 0 #0.6 0.3
      return new_x   # 0.5 0.2

    def sample(self,batch_size):
        z = self.prior.sample(batch_size)
        return self.decoder.sample(z)

    def forward(self,x):
        # x_new = self.embed.produce_latent(x)
        # run input through encoder
        x1 = self.randTrfm(x)
        x2 = self.randTrfm(x)
        #
        if self.embed_flag == True:
          z1c,z1s_mu,z1s_log_var= self.encoder.encode(self.embed.produce_latent(x1))
          z2c,z2s_mu,z2s_log_var= self.encoder.encode(self.embed.produce_latent(x2)) #(x2)
        else:
          z1c,z1s_mu,z1s_log_var= self.encoder.encode(x1)
          z2c,z2s_mu,z2s_log_var= self.encoder.encode(x2)

        z1 = self.encoder.sample(mu=z1s_mu,log_var=z1s_log_var,content_part = z1c)
        z2 = self.encoder.sample(mu=z2s_mu,log_var=z2s_log_var,content_part=z2c)
        # print(z2.shape)
        # create swap vectors
        z1s = z1[:,z1c.shape[1]:]#self.N//2
        z2s = z2[:,z2c.shape[1]:]
        z1_tilde = torch.cat((z2c,z1s),dim=1)
        z2_tilde = torch.cat((z1c,z2s),dim=1)
        # compute the swap reconstruction loss
        dec_z1_tilde = self.decoder.decode(z1_tilde)
        dec_z2_tilde = self.decoder.decode(z2_tilde)
        dec_z1 = self.decoder.decode(z1)
        dec_z2 = self.decoder.decode(z2)
        slss1 = self.rec_loss(x1,dec_z1_tilde)
        slss2 = self.rec_loss(x2,dec_z2_tilde)
        slss3 = self.rec_loss(x1,dec_z1)
        slss4 = self.rec_loss(x2,dec_z2)
        swap_loss = (slss1 +slss2 + slss3 + slss4)/4
        # compute the regularization part of the loss
        style1_kl = -0.5*(1 + z1s_log_var - z1s_mu**2 - torch.exp(z1s_log_var)).mean(-1) #sum
        style2_kl = -0.5*(1 + z2s_log_var - z2s_mu**2 - torch.exp(z2s_log_var)).mean(-1)#sum
        # compute the alignment part loss
        algn_loss = self.align_loss(z1c,z2c)
        loss_part2 = (self.beta*(style1_kl + style2_kl) + self.alpha*algn_loss).mean()
        return swap_loss + loss_part2
