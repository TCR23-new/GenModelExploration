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


class EncoderNet_SkipConn(nn.Module):
    def __init__(self,middleDim,N=78):
        super(EncoderNet_SkipConn,self).__init__()
        self.N = N
        self.middleDim = middleDim
        self.lin1 = nn.Linear(self.N,self.middleDim) # nn.Identity() self.N
        self.nrm1 = nn.BatchNorm1d(self.middleDim,eps=5e-5)
        self.lin2 = nn.Linear(self.middleDim,self.middleDim)
        self.nrm2=  nn.BatchNorm1d(self.middleDim,eps=5e-5)
        self.lin3 = nn.Linear(self.middleDim,32) #20
        self.act1 = nn.LeakyReLU(0.7) #nn.ReLU()#nn.GELU()
    def forward(self,x):
        out1 = self.act1(self.nrm1(self.lin1(x)))
        out2 = self.act1(self.nrm2(self.lin2(out1)))
        out3 = self.lin3(out2)
        return out3,out2,out1

class DecoderNet_SkipConn(nn.Module):
    def __init__(self,middleDim,N=78,valCats = None):
        super(DecoderNet_SkipConn,self).__init__()
        self.N = N
        self.middleDim = middleDim
        self.lin1 = nn.Linear(24,self.middleDim)# 15
        self.nrm1 =  nn.BatchNorm1d(self.middleDim,eps=5e-5)
        self.lin2 = nn.Linear(self.middleDim,self.middleDim)
        self.nrm2 =  nn.BatchNorm1d(self.middleDim,eps=5e-5)
        self.lin3 = nn.Linear(self.middleDim,N)#20 30 # N
        self.act1 = nn.LeakyReLU(0.7) #nn.ReLU()
        self.act2= nn.Softplus(beta=1,threshold=20)
    def forward(self,x,x2,x1):
        out1 = self.act1(self.nrm1(self.lin1(x))+x2)
        out2 = self.act1(self.nrm2(self.lin2(out1))+x1)
        out3 = self.act2(self.lin3(out2))
        return out3

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
        out,out2,out1 = self.encoder(x)
        # split the encoder output into 3 parts
        content_part = out[:,:out.shape[1]//2]
        mu,log_var = torch.chunk(out[:,out.shape[1]//2:],2,dim=1)
        return content_part,mu,log_var,[out2,out1]

    def sample(self,x=None,mu=None,log_var=None,content_part=None):
        # encode the input
        if x != None:
            content_part,mu,log_var = self.encode(x)
        # apply the reparameterization trick
        # mu,log_var = torch.chunk(torch.cat((content_part,mu,log_var),dim=1),2,dim=1)
        z = self.reparameterization(mu,log_var)
        # print('z shape {}'.format(z.shape))
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
    def decode(self,z,x2,x1):
        out = self.decoder(z,x2,x1)
        return out
    def forward(self,z,x2,x1):
        return self.decode(z,x2,x1)


class Aux_MultiOutput(nn.Module):
    def __init__(self,embed_dim,regFlag = True):
        super(Aux_MultiOutput,self).__init__()
        self.lin1 = nn.Linear(embed_dim,embed_dim)
        self.act1= nn.LeakyReLU(0.7)
        self.out_layer1 = nn.Linear(embed_dim,1)
        self.out_layer2 = nn.Linear(embed_dim,1)
        if regFlag == True:
            self.act2 = nn.ReLU()
        else:
            self.act2 = nn.Softmax(dim=1)
    def forward(self,x):
        out = self.act1(self.lin1(x))
        out1 = self.act2(self.out_layer1(out))
        out2 = self.act2(self.out_layer2(out))
        return out1,out2


class SwapVAE_ClfAux_FV(nn.Module):
    def __init__(self,encoder,decoder,N,beta,alpha,device,embed_dim,numClasses=None,numUnits=78):
        super(SwapVAE_ClfAux_FV,self).__init__()
        self.device = device
        self.encoder = Encoder(encoder,self.device,N)#,numCountVals
        self.decoder = Decoder(decoder,N=numUnits)#,numCountVals
        self.N = N
        self.rec_loss = nn.MSELoss() #nn.HuberLoss()
        self.align_loss = nn.MSELoss() #nn.HuberLoss()
        self.beta = beta
        self.alpha = alpha
        # Classification Aux component
        # self.aux_loss = nn.CrossEntropyLoss()
        # self.aux_block = nn.Sequential(
        # nn.Linear(embed_dim,embed_dim),
        # nn.LeakyReLU(0.7),
        # nn.Linear(embed_dim,numClasses),
        # nn.Softmax(dim=1)
        # )
        # Regression Aux component
        self.aux_loss = nn.MSELoss() #nn.HuberLoss()
        self.aux_block = Aux_MultiOutput(embed_dim)
    def randTrfm(self,x):
      noise = torch.normal(0,0.2,size = x.shape)
      new_x = torch.clone(x)+ noise.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
      new_x[torch.FloatTensor(x.shape).uniform_() > 0.6] = 0 #0.6 0.3
      return new_x

    def sample(self,batch_size):
        z = self.prior.sample(batch_size)
        return self.decoder.sample(z)

    def forward(self,x,y):
        # x_new = self.embed.produce_latent(x)
        # run input through encoder
        x1 = self.randTrfm(x)
        x2 = self.randTrfm(x)
        #
        z1c,z1s_mu,z1s_log_var,skip_outs1= self.encoder.encode(x1)
        z2c,z2s_mu,z2s_log_var,skip_outs2= self.encoder.encode(x2)
        z1 = self.encoder.sample(mu=z1s_mu,log_var=z1s_log_var,content_part = z1c)
        z2 = self.encoder.sample(mu=z2s_mu,log_var=z2s_log_var,content_part=z2c)
        # print(z2.shape)
        # create swap vectors
        z1s = z1[:,z1c.shape[1]:]
        z2s = z2[:,z2c.shape[1]:]
        z1_tilde = torch.cat((z2c,z1s),dim=1)
        z2_tilde = torch.cat((z1c,z2s),dim=1)
        # compute the swap reconstruction loss
        # dec_z1_tilde = self.decoder.decode(z1_tilde,skip_outs1[0],skip_outs1[1])
        # dec_z2_tilde = self.decoder.decode(z2_tilde,skip_outs2[0],skip_outs2[1])
        dec_z1 = self.decoder.decode(z1,skip_outs1[0],skip_outs1[1])
        dec_z2 = self.decoder.decode(z2,skip_outs2[0],skip_outs2[1])
        # slss1 = self.rec_loss(x1,dec_z1_tilde)
        # slss2 = self.rec_loss(x2,dec_z2_tilde)
        slss3 = self.rec_loss(x1,dec_z1)
        slss4 = self.rec_loss(x2,dec_z2)
        swap_loss = (slss3 + slss4)/2 #(slss1 +slss2 + slss3 + slss4)/4
        style1_kl = -0.5*(1 + z1s_log_var - z1s_mu**2 - torch.exp(z1s_log_var)).mean(-1) #sum
        style2_kl = -0.5*(1 + z2s_log_var - z2s_mu**2 - torch.exp(z2s_log_var)).mean(-1)#sum
        # Compute the classification loss
        aux_out11,aux_out12 = self.aux_block(z1c)
        aux_out21,aux_out22 = self.aux_block(z2c)
        aux_loss_part1 = self.aux_loss(aux_out11,y[:,0].reshape(-1,1).double()) + self.aux_loss(aux_out12,y[:,1].reshape(-1,1).double())
        aux_loss_part2 = self.aux_loss(aux_out21,y[:,0].reshape(-1,1).double()) + self.aux_loss(aux_out22,y[:,1].reshape(-1,1).double())
        aux_loss = (aux_loss_part1 + aux_loss_part2)/2
        # aux_loss = (self.aux_loss(aux_out1,y) + self.aux_loss(aux_out2,y))/2
        # compute the alignment part loss
        algn_loss = self.align_loss(z1c,z2c)
        loss_part2 = (self.beta*(style1_kl + style2_kl) + self.alpha*algn_loss).mean()
        return swap_loss + loss_part2 + aux_loss
