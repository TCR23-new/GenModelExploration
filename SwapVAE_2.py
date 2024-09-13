import numpy as np
import torch
from torch import nn

def ned_torch(x1, x2, dim=1, eps=1e-8):
    """
    Normalized eucledian distance in pytorch.

    https://discuss.pytorch.org/t/how-does-one-compute-the-normalized-euclidean-distance-similarity-in-a-numerically-stable-way-in-a-vectorized-way-in-pytorch/110829
    https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance/498753?noredirect=1#comment937825_498753

    :param x1:
    :param x2:
    :param dim:
    :param eps:
    :return:
    """
    if x1.size(1) == 1:
        ned_2 = 0.5 * ((x1 - x2)**2 / (x1**2 + x2**2 + eps))
    else:
        # num_dims = len(x1.size())
        # dim = torch.tensor(range(1, num_dims))
        ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5

class EncoderNet(nn.Module):
    def __init__(self,N,middleDim):#,numCountVals
        super(EncoderNet,self).__init__()
        self.N = N
        # self.numCountVals = numCountVals
        self.middleDim = middleDim
        self.lin1 = nn.Linear(self.N,self.middleDim)
        self.nrm1 = nn.BatchNorm1d(self.middleDim)
        self.lin2 = nn.Linear(self.middleDim,self.middleDim)
        self.nrm2= nn.BatchNorm1d(self.middleDim)
        self.lin22 = nn.Linear(self.middleDim,self.middleDim)
        self.nrm3 = nn.BatchNorm1d(self.middleDim)
        self.lin23 = nn.Linear(self.middleDim,self.middleDim)
        self.nrm4 = nn.BatchNorm1d(self.middleDim)
        self.lin24 = nn.Linear(self.middleDim,self.middleDim)
        self.nrm5 = nn.BatchNorm1d(self.middleDim)
        self.lin3 = nn.Linear(self.middleDim,44)#int(2*self.N) int(3*self.N)
        self.act1 = nn.ReLU()
    def forward(self,x):
        out1 = self.act1(self.nrm1(self.lin1(x)))
        out2 = self.act1(self.nrm2(self.lin2(out1)))
        out22 = self.act1(self.nrm3(self.lin22(out2)))
        out23 = self.act1(self.nrm4(self.lin23(out22)))
        out24 = self.act1(self.nrm5(self.lin24(out23)))
        out3 = self.lin3(out24)
        return out3

class DecoderNet(nn.Module):
    def __init__(self,N,middleDim):#,numCountVals
        super(DecoderNet,self).__init__()
        self.N = N
        # self.numCountVals = numCountVals
        self.middleDim = middleDim
        self.lin1 = nn.Linear(33,self.middleDim)#int(self.N+(self.N/2))  2*self.N
        self.nrm1 = nn.BatchNorm1d(self.middleDim)
        self.lin2 = nn.Linear(self.middleDim,self.middleDim)
        self.nrm2 = nn.BatchNorm1d(self.middleDim)
        self.lin22 = nn.Linear(self.middleDim,self.middleDim)
        self.nrm3 = nn.BatchNorm1d(self.middleDim)
        self.lin23 = nn.Linear(self.middleDim,self.middleDim)
        self.nrm4 = nn.BatchNorm1d(self.middleDim)
        self.lin24 = nn.Linear(self.middleDim,self.middleDim)
        self.nrm5 = nn.BatchNorm1d(self.middleDim)
        self.lin3 = nn.Linear(self.middleDim,int(self.N))#*self.numCountVals
        self.act1 = nn.ReLU()
        self.act2= nn.Softplus(beta=1,threshold=20)
    def forward(self,x):
        out1 = self.act1(self.nrm1(self.lin1(x)))
        out2 = self.act1(self.nrm2(self.lin2(out1)))
        out22 = self.act1(self.nrm3(self.lin22(out2)))
        out23 = self.act1(self.nrm4(self.lin23(out22)))
        out24 = self.act1(self.nrm5(self.lin24(out23)))
        out3 = self.act2(self.lin3(out24))
        return out3

class Encoder(nn.Module):
    def __init__(self,encoder_net,device,content_dim):
        super(Encoder,self).__init__()
        self.encoder = encoder_net
        self.device = device
        self.content_dim = content_dim
    def reparameterization(self,mu,log_var):
        z = mu + torch.exp(0.5*log_var)*torch.randn(log_var.shape).to(self.device)
        return z
    def encode(self,x):
        out = self.encoder(x)
        # print(out.shape)
        # split the encoder output into 3 parts
        # content_part = out[:,:self.content_dim]
        content_part = out[:,:int(out.shape[1]/2)]
        # mu,log_var = torch.chunk(out[:,self.content_dim:],2,dim=1)
        mu,log_var = torch.chunk(out[:,int(out.shape[1]/2):],2,dim=1)
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
        # reshape the output: Batch x N x numCountVals
        # out2 = out.reshape(z.shape[0],self.N)#,self.numCountVals
        return out #torch.softmax(out2,1)
    # def sample(self,z):
    #     out = self.decode(z).reshape(-1,self.numCountVals)
    #     # print(out[:5,:])
    #     out = torch.multinomial(out,1,replacement=True).reshape(z.shape[0],self.N)
    #     return out
    # def log_prob(self,x,z):
    #     # Note: THIS MAY NEED TO BE FIXED !!!!!!
    #     out = self.decode(z)
    #     m = torch.distributions.categorical.Categorical(probs=out.view(z.shape[0],self.N,self.numCountVals))#logits
    #     return m.log_prob(x).sum(-1).sum(-1)
    def forward(self,z):#,x,type="log_prob"
        # if type == "log_prob":
        #     return self.log_prob(x,z)
        # else:
        #     return self.sample(z)
        return self.decode(z)

class Prior(nn.Module):
    def __init__(self,numCountVals,device):
        super(Prior,self).__init__()
        self.numCountVals = numCountVals
        self.device = device
        self.distr = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(self.numCountVals).to(self.device), torch.eye(self.numCountVals).to(self.device))
        # MultivariateNormal(torch.zeros(2), torch.eye(2))
    def sample(self,batch_size):
        return torch.randn((batch_size,self.numCountVals))
    def log_prob(self,x):
        return self.distr.log_prob(x)

class BlockSwapVAE_N(nn.Module):
    def __init__(self,encoder,decoder,N,beta,alpha,device):#,numCountVals
        super(BlockSwapVAE_N,self).__init__()
        self.device = device
        self.encoder = Encoder(encoder,self.device,N)#,numCountVals
        self.decoder = Decoder(decoder,N)#,numCountVals
        self.N = N
        # self.numCountVals = numCountVals
        self.prior = Prior(2*self.N,self.device)
        self.rec_loss = nn.MSELoss() #nn.PoissonNLLLoss()#
        self.align_loss = ned_torch
        self.beta = beta
        self.alpha = alpha

    def randTrfm(self,x):
      new_x = torch.clone(x)
      new_x[torch.FloatTensor(x.shape).uniform_() > 0.6] = 0 #0.6
      return new_x

    def sample(self,batch_size):
        z = self.prior.sample(batch_size)
        return self.decoder.sample(z)

    def forward(self,x):
        # print(x.shape)
        # run input through encoder
        x1 = self.randTrfm(x)
        x2 = self.randTrfm(x)
        #
        z1c,z1s_mu,z1s_log_var= self.encoder.encode(x1)
        z1 = self.encoder.sample(mu=z1s_mu,log_var=z1s_log_var,content_part = z1c)
        #
        z2c,z2s_mu,z2s_log_var= self.encoder.encode(x2)
        z2 = self.encoder.sample(mu=z2s_mu,log_var=z2s_log_var,content_part=z2c)
        # create swap vectors
        # print(z1.shape)
        # print(ffdsafdsa)
        # z1s = z1[:,self.N:]#numCountVals
        # z2s = z2[:,self.N:]#.numCountVals
        z1s = z1[:,22:]#self.N numCountVals
        z2s = z2[:,22:]#self.N
        z1_tilde = torch.cat((z2c,z1s),dim=1)
        z2_tilde = torch.cat((z1c,z2s),dim=1)
        # print(z2.shape)
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
        style1_kl = -0.5*(1 + z1s_log_var - z1s_mu**2 - torch.exp(z1s_log_var)).sum(-1)
        style2_kl = -0.5*(1 + z2s_log_var - z2s_mu**2 - torch.exp(z2s_log_var)).sum(-1)
        # print('------')
        # print('------')
        # print(style1_kl)
        # print('------')
        # print(style2_kl)
        # print('------')
        # compute the alignment part loss
        algn_loss = self.align_loss(z1c,z2c)
        # print(algn_loss)
        loss_part2 = (self.beta*(style1_kl + style2_kl) + self.alpha*algn_loss).mean()
        return swap_loss + loss_part2
