from matplotlib import pyplot as plt
import utils
import loader
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from forecasting import Predictor


class Smooth_Predictor(Predictor):
    def __init__(self):
        super().__init__()
        self.method =additive_multi_exponential_smoothing
        self.model = None

    def describe(self):
        return "Exponential smoothing {}, params {}".format(self.method_str, self.params.x)

    def fit(self, series, npred=28, method='additive'):
        self.method_str = method
        if method == 'multiplicative':
            self.method = multiplicative_multi_exponential_smoothing
        else:
            self.method = additive_multi_exponential_smoothing
        self.data = (series)
        self.params = find_params(self.data, npred, self.method)
        self.predict_start = series.index[-1] + pd.Timedelta(days=1)
        res = self.method(self.data, [365, 7], self.params.x[0], self.params.x[1],self.params.x[2], [self.params.x[3], self.params.x[4]],0)
        self.resid = pd.Series(series.values  - res, series.index)



    def predict(self, npred):
        res = self.method(self.data, [365, 7], self.params.x[0], self.params.x[1],self.params.x[2], [self.params.x[3], self.params.x[4]],npred)
        predict_end = self.predict_start + pd.Timedelta(days=npred-1)
        index = pd.date_range(self.predict_start, predict_end)
        return pd.Series(data=res[-npred:], index=index)


def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen

def initial_estimates(series, ps, method = 'additive'):
    s = series[0]
    b = series[1] - series[0]
    c = []
    for p in ps:
        if method == 'additive':
            c.append(np.zeros(p))
        else:
            c.append(np.ones(p))
    return (s, b, c)

def compute_m_multiplicative(c, t, p):
        m = 1
        for i in range(len(p)):
            p_ind = t % p[i]
            if c[i][p_ind] == 0:
                continue
            m *= c[i][p_ind]
        return m

def multiplicative_multi_exponential_smoothing(series, p, alpha, beta, damp, gammas, n_preds):
    result = []
    s,b,c = initial_estimates(series, p, 'multiplicative')
    for i in range(0, len(series)+n_preds):
        if i >= len(series): # we are forecasting
            m = (i+1-len(series))
            seasonal = compute_m_multiplicative(c, i, p)
            result.append((s + m*b) * seasonal)
            try:
                if b < 0:
                    b = -np.power(-b, damp)
                else:
                    b = np.power(b, damp)
            except:
                print("b ", b, " damp " ,damp)
        else:
            x = series[i]
            m = compute_m_multiplicative(c, i, p)
            ls, s = s, alpha * (x/m) + (1 - alpha) * (s + b)
            b = beta * (s-ls) + (1-beta)*b
            for j in range(len(p)):
                if i > p[j]*3:
                    p_ind = i % p[j]
                    c[j][p_ind] = gammas[j] *( x  * c[j][p_ind] / (s* m)) + (1 - gammas[j]) * c[j][p_ind]
            result.append(s)
    return np.array(result)


def compute_m_additive(c, t, p):
    m = 0
    for i in range(len(p)):
        p_ind = t % p[i]
        m += c[i][p_ind]
    return m



def additive_multi_exponential_smoothing(series, p, alpha, beta, damp, gammas, n_preds):
    result = []
    s,b,c = initial_estimates(series, p)
    for i in range(0, len(series)+n_preds):
        if i >= len(series): # forecasting
            m = (i+1-len(series))
            seasonal = compute_m_additive(c, i, p)
            result.append(s + m*b + seasonal)
            if b < 0:
                b = -np.power(-b, damp)
            else:
                b = np.power(b, damp)
            #print("s {}, b {}, m {}, seasonal {}".format(s,b,m, seasonal))
        else:
            x = series[i]
            m = compute_m_additive(c, i, p)
            ls, s = s, alpha * (x - m) + (1 - alpha) * (s + b)
            b = beta * (s-ls) + (1-beta)*b
            for j in range(len(p)):
                p_ind = i % p[j]
                c[j][p_ind] = gammas[j] *( x - s - (m-c[j][p_ind])) + (1 - gammas[j]) * c[j][p_ind]
            result.append(s)
    return np.array(result)


def opt_fun(params, *args):
    alpha, beta, damp, gamma_year, gamma_week = params
    real_x = args[0][0]
    n_pred = args[0][1]
    method = args[0][2]
    gamma = [gamma_year, gamma_week]
    #print("params ", params)
    real_x[real_x ==0] = 1
    #res = (method(real_x, p=[365, 7], alpha=alpha, beta=beta, damp=damp, gammas=gamma,
    #              n_preds=0))
    #res[res == 0] = 1
    real_x[real_x ==0] = 1
    sse = 0
    #sse =  np.sum((real_x -  res) ** 2)/4
    real_x[real_x ==0] = 1
    for i in range(1,3):
        pred_start = len(real_x)-n_pred*i
        pred_end = len(real_x)-n_pred*(i-1)
        res = (method(real_x[pred_start:pred_end], p = [365, 7], alpha = alpha, beta= beta, damp = damp, gammas = gamma, n_preds = n_pred))
        res[res == 0] = 1
        sse += np.sum((real_x[pred_start:pred_end] - res[-n_pred:]) ** 2)
    #print("params ",params, " sse ", sse)
    return sse

def find_params(series, n_pred, method):
    initial_values = np.array([0.3, 0.1, 0.95 ,0.03,0.05])
    boundaries = [(0.01, 0.6),(0.01, 0.9), (0.8,0.95), (0.04,1), (0.05,1)]
    result = minimize(method='L-BFGS-B', fun=opt_fun,
                            x0=initial_values, args=[series, n_pred, method], bounds=boundaries)
    return result

def main():
    np.seterr(all='raise')
    data = loader.load_product_class_data("rohliky.tsv", False)
    data = data[data.index < pd.Timestamp('2017-10-01')]
    series = data.groupby(pd.Grouper(freq='D'))['product_count'].sum().fillna(0)
    series = series.astype('float')
    n_preds = 28
    predictor = Smooth_Predictor()
    predictor.fit(series[:-n_preds])
    res = predictor.predict(npred=n_preds)
    fig = plt.figure(figsize=(12, 8))
    # series.plot()
    plt.plot([i for i in range(n_preds)], res[-n_preds:].values, label='Result')
    plt.plot([i for i in range(n_preds)], series.values[-n_preds:], label='Real')
    res[series[-n_preds:].values == 0] = 0
    mape = utils.mape(series[-n_preds:], res)
    plt.legend()
    plt.show()
    print(predictor.describe(), ", mape ", mape)
    #print("params {}, mape {}, sse {}".format(params.x, utils.mape(series[-n_preds:], res[-n_preds:]), np.sum((series[-n_preds:] - res[-n_preds:]) ** 2)))

if __name__ == '__main__':
    main()