# OFC_Dataset2
# OFC_Dataset
# OFC_Dataset_Masking
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class OFC_Dataset2(Dataset):
    def __init__(self,data,scaler = MinMaxScaler()):
        self.data = data
        self.scaler = scaler
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        out ={}
        out['trialdata'] = torch.from_numpy(self.scaler.transform(np.mean(self.data[idx,:,:],axis=1).reshape(1,-1)))
        return out
    def collate_fn(self,batch):
        data = list(batch)
        trials = torch.stack([x['trialdata'] for x in data]).squeeze(1).float()
        return trials

class OFC_Dataset(Dataset):
    def __init__(self,data,scaler=None):
        self.data = data
        self.scaler = scaler #MinMaxScaler()
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        out ={}
        if self.scaler != None:
          out['trialdata'] = torch.from_numpy(self.scaler.transform(self.data[idx,:].reshape(1,-1)).squeeze(0)) # originally (self.data[idx,:,:]
        else:
          out['trialdata'] = torch.from_numpy(self.data[idx,:])
        return out
    def collate_fn(self,batch):
        data = list(batch)
        trials = torch.stack([x['trialdata'] for x in data])
        return trials

class OFC_Dataset_(Dataset):
    def __init__(self,data,value_data,scaler=None):
        self.data = data
        self.valuedata = value_data
        self.scaler = scaler #MinMaxScaler()
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        out ={}
        if self.scaler != None:
          out['trialdata'] = torch.from_numpy(self.scaler.transform(self.data[idx,:].reshape(1,-1)).squeeze(0)) # originally (self.data[idx,:,:]
        else:
          out['trialdata'] = torch.from_numpy(self.data[idx,:])
        out['value'] = self.valuedata[idx]
        return out
    def collate_fn(self,batch):
        data = list(batch)
        trials = torch.stack([x['trialdata'] for x in data])
        values = torch.tensor([x['value']-1 for x in data]).long()
        return trials,values

class OFC_Dataset2_Clf(Dataset):
    def __init__(self,data,value_data,scaler=None):
        self.data = data
        self.valuedata = value_data
        self.scaler = scaler #MinMaxScaler()
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        out ={}
        if self.scaler != None:
          out['trialdata'] = torch.from_numpy(self.scaler.transform(self.data[idx,:].reshape(1,-1)).squeeze(0)) # originally (self.data[idx,:,:]
        else:
          out['trialdata'] = torch.from_numpy(self.data[idx,:])
        out['value'] = self.valuedata[idx]
        return out
    def collate_fn(self,batch):
        data = list(batch)
        trials = torch.stack([x['trialdata'] for x in data])
        values = torch.tensor([x['value'] for x in data]).long()
        return trials,values

class OFC_Dataset2_Clf_ColorShape(Dataset):
    def __init__(self,data,value_cs_data,scaler=None):
        self.data = data
        self.valuedata = value_cs_data
        self.scaler = scaler #MinMaxScaler()
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        out ={}
        if self.scaler != None:
          out['trialdata'] = torch.from_numpy(self.scaler.transform(self.data[idx,:].reshape(1,-1)).squeeze(0)) # originally (self.data[idx,:,:]
        else:
          out['trialdata'] = torch.from_numpy(self.data[idx,:])
        out['value_c'] = self.valuedata[idx,0]
        out['value_s'] = self.valuedata[idx,1]
        return out
    def collate_fn(self,batch):
        data = list(batch)
        trials = torch.stack([x['trialdata'] for x in data])
        values_c = torch.tensor([x['value_c'] for x in data]).long()
        values_s = torch.tensor([x['value_s'] for x in data]).long()
        return trials,values_c,values_s

class OFC_Dataset_wMasking(Dataset):
    def __init__(self,data,maskRatio=0.2):
        self.data = data
        self.maskRatio = maskRatio
        self.scaler = MinMaxScaler()
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        out ={}
        out['trialdata'] = torch.from_numpy(self.data[idx,:,:]) #torch.from_numpy(self.scaler.fit_transform(self.data[idx,:,:]))
        mask_ids = np.random.choice(range(self.data.shape[2]), size=int(0.2*self.data.shape[2]), replace=False)
        mask_trial_data = out['trialdata']
        mask_trial_data[:,mask_ids] = 0
        out['maskdata'] =  mask_trial_data
        return out
    def collate_fn(self,batch):
        data = list(batch)
        trials = torch.stack([x['trialdata'] for x in data])
        mask_trials = torch.stack([x['maskdata'] for x in data])
        return trials,mask_trials
