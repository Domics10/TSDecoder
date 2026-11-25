"""Errori di una previsione rispetto ai valori osservati. Solo numpy.

Rispetto alla prima versione: niente statsmodels (l'autocorrelazione si calcola qui), nessuna divisione
per zero (una metrica che non si può calcolare vale None invece di inf o nan), e in più sMAPE e MASE.
"""
import numpy as np


def _finite(value):
    """Il numero come float, o None se non è finito."""
    return float(value) if value is not None and np.isfinite(value) else None


class ForecastErrors:
    """Errori fra `forecast` e `actual` (stessa lunghezza, almeno un punto).

    `history` è la serie usata per prevedere: serve al MASE, che confronta l'errore con quello di una
    previsione naive sullo storico (ripetere il valore di `season` passi prima).
    """

    def __init__(self, forecast, actual, history=None, season=1):
        self.forecast = np.asarray(forecast, dtype=float)
        self.actual = np.asarray(actual, dtype=float)
        if self.forecast.shape != self.actual.shape or self.forecast.ndim != 1 or self.forecast.size == 0:
            raise ValueError("forecast e actual devono essere liste della stessa lunghezza, non vuote.")
        self.errors = self.forecast - self.actual
        self.history = None if history is None else np.asarray(history, dtype=float)
        self.season = max(1, int(season))

    def me(self):
        """Errore medio (bias): positivo se la previsione sta sopra."""
        return _finite(np.mean(self.errors))

    def mae(self):
        return _finite(np.mean(np.abs(self.errors)))

    def rmse(self):
        return _finite(np.sqrt(np.mean(self.errors ** 2)))

    def mape(self):
        """Errore percentuale assoluto medio, sui soli punti con valore osservato diverso da zero."""
        mask = self.actual != 0
        return _finite(np.mean(np.abs(self.errors[mask] / self.actual[mask]))) if mask.any() else None

    def mpe(self):
        mask = self.actual != 0
        return _finite(np.mean(self.errors[mask] / self.actual[mask])) if mask.any() else None

    def smape(self):
        """MAPE simmetrico (0–2): regge anche con valori osservati vicini a zero."""
        denominator = np.abs(self.forecast) + np.abs(self.actual)
        mask = denominator != 0
        return _finite(np.mean(2 * np.abs(self.errors[mask]) / denominator[mask])) if mask.any() else None

    def mase(self):
        """MAE diviso l'errore della previsione naive sullo storico: sotto 1 batte il metodo naive."""
        s = self.season
        if self.history is None or self.history.size <= s:
            return None
        scale = np.mean(np.abs(self.history[s:] - self.history[:-s]))
        return _finite(self.mae() / scale) if scale > 0 else None

    def corr(self):
        """Correlazione fra previsti e osservati; None se una delle due serie è costante."""
        if self.forecast.size < 2 or np.std(self.forecast) == 0 or np.std(self.actual) == 0:
            return None
        return _finite(np.corrcoef(self.forecast, self.actual)[0, 1])

    def minmax(self):
        """1 - media(min/max) punto per punto; ha senso con valori positivi."""
        lows = np.minimum(self.forecast, self.actual)
        highs = np.maximum(self.forecast, self.actual)
        return _finite(1 - np.mean(lows / highs)) if np.all(highs > 0) else None

    def acf1(self):
        """Autocorrelazione a ritardo 1 degli errori: vicina a 0 se gli errori non hanno una struttura."""
        centered = self.errors - np.mean(self.errors)
        denominator = np.sum(centered ** 2)
        if centered.size < 3 or denominator == 0:
            return None
        return _finite(np.sum(centered[1:] * centered[:-1]) / denominator)

    METRICS = ("me", "mae", "rmse", "mape", "mpe", "smape", "mase", "corr", "minmax", "acf1")

    def summary(self):
        return {name: getattr(self, name)() for name in self.METRICS} | {"points": int(self.errors.size)}


def forecast_accuracy(forecast, actual):
    """Compatibilità con main.py: le metriche della prima versione, con gli stessi nomi."""
    errors = ForecastErrors(forecast, actual)
    return {
        "mape": errors.mape(), "me": errors.me(), "mae": errors.mae(), "mpe": errors.mpe(),
        "rmse": errors.rmse(), "corr": errors.corr(), "minmax": errors.minmax(), "acfe": errors.acf1(),
    }
