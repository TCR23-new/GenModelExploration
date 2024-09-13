# Scaling Net
# Translation Net
# NVP_TrialData
import numpy as np
import torch
from torch import nn

class ScalingNet(nn.Module):
    def __init__(self,shape,middleDim):
        super(ScalingNet,self).__init__()
        self.shape = shape
        self.middleDim = middleDim
        self.linear1 = nn.Linear(self.shape//2,middleDim)
        self.conv_block = nn.Sequential(
            nn.Conv1d(1,1,kernel_size=6,stride=2),
            nn.Conv1d(1,1,kernel_size=6,stride=2),
            nn.Conv1d(1,1,kernel_size=6,stride=1),
        )
        conv_out_shape = self._get_conv_shape()
        self.linear2 = nn.Linear(conv_out_shape,self.shape//2)
        self.act1 = nn.ReLU()#Leaky
        self.act2 = nn.Tanh()
    def _get_conv_shape(self):
      input = torch.zeros(1,1,self.middleDim)
      out = self.conv_block(input)
      return out.shape[2]
    def forward(self,x):
        out1 = self.act1(self.linear1(x))
        out12 = self.act1(self.conv_block(out1.unsqueeze(1)))
        out2 = self.act1(self.linear2(out12.squeeze(1)))
        return out2

class TranslationNet(nn.Module):
    def __init__(self,shape,middleDim):
        super(TranslationNet,self).__init__()
        self.shape = shape
        self.middleDim = middleDim
        self.linear1 = nn.Linear(self.shape//2,middleDim)
        self.conv_block = nn.Sequential(
            nn.Conv1d(1,1,kernel_size=6,stride=2),
            nn.Conv1d(1,1,kernel_size=6,stride=2),
            nn.Conv1d(1,1,kernel_size=6,stride=1),
        )
        conv_out_shape = self._get_conv_shape()
        self.linear2 = nn.Linear(conv_out_shape,self.shape//2)
        self.act1 = nn.ReLU()#Leaky
        self.act2 = nn.Tanh()
    def _get_conv_shape(self):
      input = torch.zeros(1,1,self.middleDim)
      out = self.conv_block(input)
      return out.shape[2]
    def forward(self,x):
        out1 = self.act1(self.linear1(x))
        out12 = self.act1(self.conv_block(out1.unsqueeze(1)))
        out2 = self.linear2(out12.squeeze(1))
        return out2

class NVP_TrialData(nn.Module):
    def __init__(self,s_net,t_net,input_shape,prior,device,numOfFlows = 5):
        super(NVP_TrialData,self).__init__()
        self.numOfFlows = numOfFlows
        self.device = device
        self.input_shape = input_shape
        # create the collection of nets for the flow
        self.t_nets = nn.ModuleList([t_net for _ in range(numOfFlows)])
        self.s_nets = nn.ModuleList([s_net for _ in range(numOfFlows)])
        self.prior = prior
    def coupling(self,x,index,forward_pass=True):
        # split the input
        [xa,xb] = x.chunk(2,dim=1)#,
        # perform the processing
        s = self.s_nets[index](xa.to(self.device))
        t = self.t_nets[index](xa.to(self.device))
        # determine correct processing based on if we are going x->z or vice versa
        if forward_pass:
            ya = xa
            yb = (xb-t)*torch.exp(-s)
        else:
            ya = xa
            yb = torch.exp(s)*xb + t
        return torch.cat((ya,yb),dim=1).to(self.device),s

    def permutation(self,x):
        return torch.flip(x,[1])

    def f(self,x):
        log_J_det = torch.zeros(x.shape[0]).to(self.device)
        z = x
        for i in range(self.numOfFlows):
            z,s = self.coupling(z,i)
            z = self.permutation(z)
            log_J_det -= s.sum(dim=1)
        return log_J_det,z

    def f_inv(self,z):
        x = z
        for i in range(self.numOfFlows):
            x = self.permutation(x)
            x,_ = self.coupling(x,i,forward_pass = False)
        return x

    def sample(self,numOfSamples):
        # Create the variable to store the samples
        # self.samples = torch.zeros((numOfSamples,self.input_shape[0],self.input_shape[1]))
        #
        z = self.prior.sample((numOfSamples,self.input_shape))[:,0,:]
        # process the latent variable
        x= self.f_inv(z).reshape(-1,self.input_shape)
        return x

    def forward(self,x,reduction = "avg"):
        log_J_det,z = self.f(x)
        if reduction == "sum":
            return -(self.prior.log_prob(z).to(self.device) + log_J_det.to(self.device)).sum()
        else:
            a11 = self.prior.log_prob(z).to(self.device)
            a12 = log_J_det.to(self.device)
            return -(a11+ a12).mean()
