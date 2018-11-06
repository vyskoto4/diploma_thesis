from collections import Counter

from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, EasterMonday, GoodFriday
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

SCALER = MinMaxScaler()


class CzechHolidayCalendar(AbstractHolidayCalendar):
    """
        Czech holidays calendar
            """
    rules = [
        Holiday('New Years Day', month=1, day=1),
        EasterMonday,
        GoodFriday,
        Holiday('Labor Day', month=5, day=1),
        Holiday('Victory Day', month=5, day=8),
        Holiday('St Cyril and Methodius Day', month=7, day=5),
        Holiday('Jan Hus Day', month=7, day=6),
        Holiday('Czech Statehood Day', month=9, day=28),
        Holiday('Foundation of Czechoslovak state', month=10, day=28),
        Holiday('Freedom and Democracy Day', month=11, day=17),
        Holiday('Christmas Eve', month=12, day=24),
        Holiday('Christmas Day', month=12, day=25),
        Holiday('St Stephens Day', month=12, day=26)
    ]

def remove_holidays(series):
    vals = series.copy()
    cal = CzechHolidayCalendar()
    hols = cal.holidays(series.index[0], series.index[-1])
    mean = series.rolling(window=7).mean()
    mean = mean.fillna(0)
    vals[hols] = mean[hols]
    return vals

def train_test_split(x, y, test_samples):
    return np.array(x[:-test_samples, ]), np.array(y[:-test_samples, ]), np.array(x[-test_samples:, ]), np.array(
        y[-test_samples:, ])

def mape(y_true, y_pred):
    true = np.array(y_true.values)
    pred = np.array(y_pred.values)
    pred[(true == 0) * (pred != 0)] += 1
    pred[(true == 0) * (pred == 0)] = 1
    true[true == 0] = 1
    return np.mean(np.abs((true - pred) / true)) * 100

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))

def vals_to_pdf(vals):
    c = Counter(vals)
    all = sum(c.values())*1.0
    ret = {}
    for k,v in c.items():
        ret[k] = v/all
    return ret


def to_list(var, len):
    if type(var) is list or type(var) is np.ndarray:
        return var
    else:
        return [var]*len


def scale_series(ser):
    arr = np.array(ser.values)
    real_y = SCALER.fit_transform(arr.reshape(-1,1)).flatten()
    return pd.Series(data=real_y, index = ser.index)

def scale_by_max(ser):
    ret1 = ser/get_max(ser)
    return ret1

def get_max(ser):
    return  1.0*ser.max()
