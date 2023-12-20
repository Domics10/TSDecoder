import utils
import seaborn as sns
import model
import matplotlib.pyplot as plt
import torch
import pandas as pd

def split_dataframe(data, train_ratio=0.8, val_ratio=0.2):
    total_length = len(data)
    train_length = int(total_length * train_ratio)
    val_length = int(total_length * val_ratio)
    test_length = total_length - train_length - val_length

    train_data = data.iloc[:train_length]
    val_data = data.iloc[train_length:train_length + val_length]
    test_data = data.iloc[train_length + val_length:]

    return train_data, val_data, test_data

def main():
    csv = utils.load_to_dataframe('./data/air-temp-monthly-mean.csv')
    csv.rename(columns={"month":"ds", "mean_temp":"y"}, inplace=True)
    train_data, val_data, _ = split_dataframe(csv)
    #training(train_data, val_data)
    forecasting(train_data, val_data)

def training(train_data, val_data):
    loss_over_epoch = model.decoder_training_pipeline(train_data, val_data)
    sns.set_style('darkgrid')
    fig, _ = plt.subplots(figsize=(12, 6), dpi=100)
    plt.plot(loss_over_epoch['epoch'], loss_over_epoch['t_loss'], label='T loss')
    plt.plot(loss_over_epoch['epoch'], loss_over_epoch['v_loss'], label='V loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    fig.savefig(f'./saved_models/loss_over_epochs.jpg')

def forecasting(train_data, val_data):
    starting_window = torch.tensor(train_data['y'][-90:].values, dtype=torch.float)
    forecast = model.decoder_forecasting_pipeline(starting_window, len(val_data))
    forecast = pd.Series(forecast.squeeze())
    val_data.loc[:, 'yhat'] = forecast.values
    metrics = pd.DataFrame(model.forecast_accuracy(val_data['yhat'].values, val_data['y'].values), index=[0])
    utils.instance_path.set_path(f'./plots/',f'forecast_')
    utils.decoder_plotter(val_data, False, metrics).plot_helper()

if __name__ == '__main__':
    main()