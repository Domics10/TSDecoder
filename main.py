"""Validazione del decoder sui pesi salvati; con --addestra prima lo addestra (e sovrascrive i pesi).

    python main.py             # validazione: grafico e metriche in plots/
    python main.py --addestra  # addestramento (loss in saved_models/), poi validazione
"""
import argparse
import utils
from utils import plots
from model.decoder import decoder_training_pipeline, decoder_forecasting_pipeline
from model.metrics import forecast_accuracy
import torch
import pandas as pd

def split_dataframe(data, train_ratio=0.8):
    train_length = int(len(data) * train_ratio)
    return data.iloc[:train_length], data.iloc[train_length:]

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--addestra', action='store_true', help='addestra prima di validare (sovrascrive i pesi salvati)')
    args = parser.parse_args()
    csv = utils.load_to_dataframe('./data/air-temp-monthly-mean.csv')
    csv.rename(columns={"month":"ds", "mean_temp":"y"}, inplace=True)
    train_data, val_data = split_dataframe(csv)
    if args.addestra:
        training(train_data, val_data)
    forecasting(train_data, val_data)

def training(train_data, val_data):
    loss_over_epoch = decoder_training_pipeline(train_data, val_data)
    plots.plot_loss(loss_over_epoch, './saved_models/loss_over_epochs.jpg')

def forecasting(train_data, val_data):
    window = utils.utils().get_d_model()
    starting_window = torch.tensor(train_data['y'][-window:].values, dtype=torch.float)
    forecast = decoder_forecasting_pipeline(starting_window, len(val_data))
    val_data = val_data.assign(yhat=pd.Series(forecast.squeeze()).values)
    metrics = pd.DataFrame(forecast_accuracy(val_data['yhat'].values, val_data['y'].values), index=[0])
    plots.plot_validation(val_data, metrics, plots.create_folder_with_timestamp('./plots', 'forecast'))

if __name__ == '__main__':
    main()
