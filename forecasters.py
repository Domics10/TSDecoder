"""Forecaster registrati per nome.

Ogni forecaster riceve la lista dei valori e l'orizzonte e restituisce i valori
previsti; `min_points` dice quanti punti servono almeno in ingresso.
Un modello nuovo si aggiunge con una classe e una voce in `build_registry`.
"""
import numpy as np
import pandas as pd


def rescale(values, src, dst):
    """Porta linearmente `values` dall'intervallo `src` all'intervallo `dst`."""
    (src_lo, src_hi), (dst_lo, dst_hi) = src, dst
    src_span = src_hi - src_lo
    if src_span == 0:
        return np.full(len(values), (dst_lo + dst_hi) / 2.0)
    return dst_lo + (np.asarray(values, dtype=float) - src_lo) * (dst_hi - dst_lo) / src_span


class LinearForecaster:
    """Regressione lineare sull'indice: il modello di prova, senza requisiti."""

    min_points = 2

    def __call__(self, values, horizon):
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        future = np.arange(len(values), len(values) + horizon)
        return (slope * future + intercept).tolist()


class TSDecoderForecaster:
    """Il decoder addestrato in questo repo, usato in sola inferenza.

    Il modello lavora su una finestra di `d_model` punti nella scala dei dati di
    addestramento: la finestra in ingresso si riporta in quella scala e le
    previsioni si riportano indietro.
    """

    def __init__(self, decoder, forecaster_cls, window, train_range):
        self.decoder = decoder
        self.forecaster_cls = forecaster_cls
        self.min_points = window
        self.train_range = train_range

    @classmethod
    def load(cls, train_csv="./data/air-temp-monthly-mean.csv", train_ratio=0.8):
        import torch  # import pesante: solo se il modello si carica
        import utils
        from model.decoder import Decoder, Forecaster

        config = utils.utils()
        config.rename("tsdecoder")
        decoder = config.load(Decoder(config.get_heads(), config.get_d_model(), c1=config.get_c1()))
        decoder.eval()

        series = utils.load_to_dataframe(train_csv).iloc[:, 1]
        train = series.iloc[: int(len(series) * train_ratio)]
        return cls(decoder, Forecaster, config.get_d_model(), (train.min(), train.max()))

    def __call__(self, values, horizon):
        import torch

        window = np.asarray(values[-self.min_points:], dtype=float)
        input_range = (window.min(), window.max())
        scaled = rescale(window, input_range, self.train_range)
        with torch.no_grad():
            forecast = self.forecaster_cls(self.decoder)(
                torch.tensor(scaled, dtype=torch.float).unsqueeze(0), horizon
            )
        return rescale(forecast.squeeze(0).numpy(), self.train_range, input_range).tolist()


def build_registry():
    return {
        "linear": LinearForecaster(),
        "TSDecoder": TSDecoderForecaster.load(),
    }


def next_timestamps(timestamps, horizon):
    """Timestamp successivi all'ultimo, con il passo della serie in ingresso."""
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    freq = pd.infer_freq(index) if len(index) >= 3 else None
    step = freq or (index[1:] - index[:-1]).median()
    future = pd.date_range(start=index[-1], periods=horizon + 1, freq=step)[1:]
    return [ts.isoformat().replace("+00:00", "Z") for ts in future]
