import numpy as np
from statsmodels.tsa.stattools import acf

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast-actual)/np.abs(actual))
    me = np.mean(forecast-actual)
    mae = np.mean(np.abs(forecast-actual))
    mpe = np.mean((forecast-actual)/actual)
    rmse = np.mean((forecast-actual)**2)**0.5
    corr = np.corrcoef(forecast, actual)[0,1]
    mins = np.amin(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)
    acfe = acf(forecast - actual)[1]
    return ({'mape':mape, 'me':me, 'mae':mae, 'mpe':mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax, 'acfe':acfe})