import pandas as pd
from fbprophet import Prophet

from forecasting import Predictor
from utils import CzechHolidayCalendar


class Prophet_Predictor(Predictor):
    def __init__(self):
        super().__init__()
        self.model = None

    def describe(self):
        return "Prophet"

    def fit(self, series, npred = 28):
        train = series.reset_index()
        train.columns = ['ds', 'y']
        self.model = Prophet(interval_width=0.95, holidays = create_holidays())
        self.model.fit(train)
        forecast = self.model.predict()
        df = pd.merge(train, forecast, on='ds')
        self.resid =  df['y'] - df['yhat']
        self.resid.index = series.index
        self.predict_start = series.index[-1] + pd.Timedelta(days=1)


    def predict(self, npred):
        future = self.model.make_future_dataframe(periods=npred)
        forecast = self.model.predict(future)
        predict_end = self.predict_start + pd.Timedelta(days=npred - 1)
        index = pd.date_range(self.predict_start, predict_end)
        return pd.Series(data= forecast['yhat'][-npred:].values, index=index)

def create_holidays():
    holiday_names = ['new years day', 'good friday', 'easter monday', 'labor day', 'victory day', 'cyril and methodus', ' jan hus day',
                 'czech statehood', 'foundation of czechoslovak state', 'freedom and democcracy day', 'christmas eve',
                 'christmas day', 'st stephens day']
    cal = CzechHolidayCalendar()
    hols = cal.holidays(pd.Timestamp('2014-01-01'), pd.Timestamp('2018-12-31'))
    names = int(len(hols)/len(holiday_names))*holiday_names
    df = pd.DataFrame({'holiday': names, 'ds': hols})
    return df
