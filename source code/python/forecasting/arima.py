import sys

import statsmodels.api as sm
import itertools
from forecasting import Predictor


class ARIMA_Predictor(Predictor):

    def __init__(self, criterion = 'bic'):
        super().__init__()
        self.model = None
        self.order = None
        self.seasonal_order = None
        self.resid = None
        self.series_len = None
        self.trend = None
        self.criterion = criterion

    def describe(self):
        return "ARIMA order {}, seasonal order {}".format(self.order, self.seasonal_order)

    def fit(self, series, npred = 28):
        best_fits = auto_arima(series=series)
        order, seasonal_order, trend = best_fits[self.criterion][1]
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model = sm.tsa.statespace.SARIMAX(series, order = order, trend=trend, seasonal_order =seasonal_order ,enforce_invertibility = False ,enforce_stationarity=False).fit()
        self.resid = self.model.resid
        self.series_len = len(series)

    def predict(self, npred):
        pred_start = self.series_len
        pred_end = self.series_len + npred-1
        prediction = self.model.get_prediction(pred_start, pred_end)
        return prediction.predicted_mean


def auto_arima(series):
    p = d = q = range(0, 3)
    s =[7]
    best_result = {'aic':[1000000,None], 'bic':[1000000,None]}
    orders = list(itertools.product(p, d, q))
    trends = ['c','ct']
    p = d = q = range(0, 2)
    seasonal_orders = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q, s))]
    train = series
    for order in orders:
        for seasonal_order in seasonal_orders:
            for trend in trends:
                try:
                    model = sm.tsa.statespace.SARIMAX(train, order = order, trend=trend, seasonal_order =seasonal_order ,enforce_invertibility = False ,enforce_stationarity=False)
                    model_fit = model.fit()
                    print("Fit ARIMA, order = {}, seasonal_oder = {}, trend ={}, mape {}, AIC = {}".format(order, seasonal_order, trend, 0, model_fit.aic))
                    for k,v in best_result.items():
                        if k == 'aic':
                            result = model_fit.aic
                        elif k == 'bic':
                            result = model_fit.bic
                        if v[0] > result:
                            v[0] = result
                            v[1] = [order, seasonal_order, trend]
                except ValueError:
                    print("Error while fitting ARIMA, order = {}, seasonal_oder = {}, trend ={}, error: ".format(order, seasonal_order, trend, sys.exc_info()[0]))
    return best_result
