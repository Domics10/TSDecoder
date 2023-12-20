import os
import time
import json
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np

class instance_path:
    path = None
    @staticmethod
    def set_path(path, folder):
        if not os.path.exists(path):
            os.makedirs(path)
        instance_path.path = create_folder_with_timestamp(f'{path}/{folder}')
    @staticmethod
    def get_path():
        return instance_path.path

def create_folder_with_timestamp(name):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    folder_name = f"{name}_{timestamp}"
    os.mkdir(folder_name)
    return folder_name

class utils:
    def __init__(self):
        json_obj = self.load_json_model()
        self.epochs = json_obj['epochs']
        self.heads = json_obj['heads']
        self.c1 = json_obj['c1']
        self.path = json_obj['path']
        self.name = json_obj['name']+'.pth'
        self.d_model = json_obj['d_model']
        self.model = None

    def rename(self, name):
        self.name = 'epochs_'+str(name)+'.pth'
    
    def get_name(self):
        return self.name

    def get_epochs(self):
        return self.epochs
    
    def get_c1(self):
        return self.c1
    
    def get_heads(self):
        return self.heads
    
    def get_d_model(self):
        return self.d_model

    def load_json_model(self):
        with open('./model/model.json') as file:
            json_obj = json.load(file)
        return json_obj['custom_hyper_parameters']['tsdecoder']

    def save(self, model):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(model.state_dict(), self.path+self.name)

    def load(self, model):
        self.model = model
        self.model.load_state_dict(torch.load(self.path+self.name))
        return self.model
    
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
    
class forecast_helper:
    def __init__(self, model_name, future_flag, dataframe):
        self.model_name = model_name
        self.future_flag = future_flag
        self.dataframe = dataframe

    def plot_helper(self):
        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=(20, 6), dpi=200)
        ax.figure.autofmt_xdate()
        ax.set_xticks(np.arange(0, len(self.dataframe), 5))
        fig.autofmt_xdate()
        return fig

    def save_helper(self, fig, metrics):
        namefolder = instance_path.get_path()
        if self.future_flag:
            fig.savefig(f'{namefolder}/forecast_{self.model_name}.jpg')
        else:
            fig.savefig(f'{namefolder}/validation_{self.model_name}.jpg')
            metrics.to_csv(f'{namefolder}/metrics_{self.model_name}.csv', index=False)

class decoder_plotter(forecast_helper):
    def __init__(self, dataframe, future_flag, metrics):
        super().__init__('provisory_name',future_flag, dataframe)
        self.dataframe = dataframe
        self.future_flag = future_flag
        self.metrics = metrics
        self.fig = None
    
    def save_helper(func):
        def wrapper(self):
            func(self)
            super().save_helper(self.fig, self.metrics)
        return wrapper

    @save_helper
    def plot_helper(self):
        fig = super().plot_helper()
        if self.future_flag:
            pass
            #plt.plot(self.dftest['ds'], self.forecast['y'], color='blue', label='forecast')
        else :
            plt.plot(self.dataframe['ds'], self.dataframe['y'], '.', color='orange', label='Validation')
            plt.plot(self.dataframe['ds'], self.dataframe['yhat'], color='blue', label='Forecast')
        plt.legend(loc='upper right')
        self.fig = fig

def generate_future_dates(dataframe, date_column, periods):
    freq = pd.infer_freq(dataframe[date_column])
    last_date = dataframe[date_column].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq=freq)
    future_df = pd.DataFrame({date_column: future_dates})
    return future_df

def load_to_dataframe(path : str) -> pd.DataFrame :
    if '.xlsx' in path :
        return pd.read_excel(path)
    else :
        return pd.read_csv(path, sep=';')