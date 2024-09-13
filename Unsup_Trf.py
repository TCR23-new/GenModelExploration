# OFC_Data_Transformer_Unsup
import numpy as np
import torch
from torch import nn

class OFC_Data_Transformer_Unsup(nn.Module):
  def __init__(self,channelNum,embed_dim):
      super(OFC_Data_Transformer_Unsup,self).__init__()
      self.pos_enc = nn.Parameter(torch.randn(channelNum,1,embed_dim))
      self.cls_head = nn.Parameter(torch.randn(1,1, embed_dim))
      self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=embed_dim,nhead=1,dropout=0.5,activation='relu')
      self.trf = nn.TransformerEncoder(self.encoder_layer1,6)
      self.linear = nn.Linear(embed_dim,embed_dim)
  def forward(self,x):
    # Assuming shape BxNxT
    out42 = x.permute(2,0,1) # TxBxN
    out42 = out42 + self.pos_enc
    trf_out = self.linear(self.trf(out42))
    return torch.exp(trf_out.permute(1,2,0))
