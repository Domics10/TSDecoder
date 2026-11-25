import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset

class utils:
    """Iperparametri di model/model.json e salvataggio/caricamento dei pesi."""
    def __init__(self):
        json_obj = self.load_json_model()
        self.epochs = json_obj['epochs']
        self.heads = json_obj['heads']
        self.c1 = json_obj['c1']
        self.lr = json_obj['lr']
        self.batch_size = json_obj['batch_size']
        self.path = json_obj['path']
        self.name = 'epochs_'+json_obj['name']+'.pth'
        self.d_model = json_obj['d_model']

    def get_epochs(self):
        return self.epochs

    def get_c1(self):
        return self.c1

    def get_heads(self):
        return self.heads

    def get_d_model(self):
        return self.d_model

    def get_lr(self):
        return self.lr

    def get_batch_size(self):
        return self.batch_size

    def load_json_model(self):
        with open('./model/model.json') as file:
            json_obj = json.load(file)
        return json_obj['custom_hyper_parameters']['tsdecoder']

    def save(self, model):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(model.state_dict(), self.path+self.name)

    def load(self, model):
        model.load_state_dict(torch.load(self.path+self.name, map_location='cpu', weights_only=True))
        return model

class TimeSeriesData(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        input_sequence = self.data['y'].iloc[idx:idx + self.sequence_length].values
        target = self.data['y'].iloc[idx + self.sequence_length]
        return torch.tensor(input_sequence, dtype=torch.float), torch.tensor(target, dtype=torch.float)

def load_to_dataframe(path : str) -> pd.DataFrame :
    if '.xlsx' in path :
        return pd.read_excel(path)
    else :
        return pd.read_csv(path, sep=';')
