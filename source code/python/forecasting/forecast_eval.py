import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import configuration
import loader
import utils
from forecasting.arima import ARIMA_Predictor
from forecasting.exp_smooth import Smooth_Predictor
from forecasting.prophet import Prophet_Predictor

predictors = {'Smoothing': Smooth_Predictor(),
              'Prophet': Prophet_Predictor(),
              'ARIMA': ARIMA_Predictor(),
              #'LSTM': LSTM_Predictor()
              }


def train_test_split(series, n_preds=configuration.N_PREDS):
    train = series[:-n_preds]
    test = series[-n_preds:]
    return train,test

def remove_if_needed(path):
    if os.path.isfile(path):
        os.remove(path)

def save_result(results,model_name, sku, period_ind, mape, rmse, desc):
    results[model_name][sku][period_ind]['mape'] = mape
    results[model_name][sku][period_ind]['rmse'] = rmse
    results[model_name][sku][period_ind]['desc'] = desc
    fname = configuration.FORECAST_RES_DIR+'results.json'
    remove_if_needed(fname)
    with open(fname, 'w') as outfile:
        json.dump(results, outfile,indent=4)

def save_plot(test_scaled, res_scaled, end_of_period, res_path):
    fig, ax = plt.subplots()
    test_scaled.plot(label='Actual value')
    res_scaled.plot(label='Forecast')
    ax.set_xlim(end_of_period - pd.Timedelta(days=configuration.N_PREDS + 1), end_of_period + pd.Timedelta(days=3))
    plt.xlabel('Day of month')
    plt.legend()
    plt.ylabel('Sold units [Scaled]')
    img = res_path + ".pdf"
    remove_if_needed(img)
    plt.tight_layout()
    plt.savefig(img)
    plt.close(fig)

def save_forecast_resid(forecast, resid, res_path):
    forecast_pickle = res_path + "_forecast.pickle"
    resid_pickle = res_path + "_resid.pickle"
    remove_if_needed(forecast_pickle)
    remove_if_needed(resid_pickle)
    forecast.to_pickle(forecast_pickle)
    resid.to_pickle(resid_pickle)


def init_results():
    fname = configuration.FORECAST_RES_DIR+'results.json'
    if os.path.isfile(fname):
        with open(fname) as f:
            results = json.load(f)
    else:
        results = {'Smoothing': defaultdict(str),
                   'Prophet': defaultdict(str),
                   'ARIMA': defaultdict(str),
                   'LSTM': defaultdict(str)}
    for model_name, p in results.items():
        for sku in configuration.SKUS:
            if not sku in results[model_name]:
                results[model_name][sku] = [{'mape':np.inf, 'rmse': np.inf, 'desc': ""} for i in range(0,len(configuration.PERIODS))]
    return results

def main():
    results = init_results()
    for model_name, predictor in predictors.items():
        for sku in configuration.SKUS:
            for period_ind in range(len(configuration.PERIODS)):
                period = configuration.PERIODS[period_ind]
                res_path = configuration.FORECAST_RES_DIR+model_name+"\\"+sku+"\\"+str(period_ind)
                end_of_period = period[1]
                real_series = loader.load_test_sku(sku, base_dir=configuration.BASE_DIR, end_of_period=end_of_period)
                train, test = train_test_split(real_series, configuration.N_PREDS)
                train = utils.remove_holidays(train)
                predictor.fit(train, configuration.N_PREDS)
                forecast = predictor.predict(configuration.N_PREDS)
                resid = predictor.resid
                forecast_scaled = utils.scale_by_max(forecast)
                test_scaled = utils.scale_by_max(test)
                save_plot(test_scaled, forecast_scaled, end_of_period, res_path)
                save_forecast_resid(forecast, resid, res_path)
                mape = utils.mape(y_true=test, y_pred=forecast)
                rmse = utils.rmse(y_true=test_scaled, y_pred=forecast_scaled)
                save_result(results, model_name, sku, period_ind,mape, rmse, predictor.describe())

if __name__ == '__main__':
    main()