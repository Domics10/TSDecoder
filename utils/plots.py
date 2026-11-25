"""Grafici dell'addestramento e della validazione: li usa solo main.py, non il servizio di previsione."""
import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_folder_with_timestamp(root, prefix):
    """Crea e restituisce la cartella `root/prefix_<data-ora>`, dove finiscono i file di una validazione."""
    os.makedirs(root, exist_ok=True)
    folder = os.path.join(root, f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.mkdir(folder)
    return folder

def plot_validation(dataframe, metrics, folder, model_name='tsdecoder'):
    """Valori di validazione e previsione (colonne ds, y, yhat) nel grafico; metriche in CSV."""
    sns.set_style('darkgrid')
    fig, ax = plt.subplots(figsize=(20, 6), dpi=200)
    ax.set_xticks(np.arange(0, len(dataframe), 5))
    fig.autofmt_xdate()
    ax.plot(dataframe['ds'], dataframe['y'], '.', color='orange', label='Validation')
    ax.plot(dataframe['ds'], dataframe['yhat'], color='blue', label='Forecast')
    ax.legend(loc='upper right')
    fig.savefig(f'{folder}/validation_{model_name}.jpg')
    metrics.to_csv(f'{folder}/metrics_{model_name}.csv', index=False)
    plt.close(fig)

def plot_loss(loss_over_epoch, path):
    """Loss di addestramento e di validazione per epoca."""
    sns.set_style('darkgrid')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    ax.plot(loss_over_epoch['epoch'], loss_over_epoch['t_loss'], label='T loss')
    ax.plot(loss_over_epoch['epoch'], loss_over_epoch['v_loss'], label='V loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Loss over Epochs')
    fig.savefig(path)
    plt.close(fig)
