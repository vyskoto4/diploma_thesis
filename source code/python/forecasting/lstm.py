import utils
from forecasting import Predictor
import loader
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM

class LSTM_Predictor(Predictor):
    def __init__(self):
        super().__init__()
        self.model = None

    def describe(self):
        return "LSTM"

    def prepare_data(self, series, back = 7):
        back = 7
        xvals = []
        for i in range(1, back + 1):
            xvals.append(series.shift(i).fillna(0))
        nn_x = np.vstack(xvals)
        nn_x = nn_x.transpose()
        nn_y = np.array(series.values)
        # split data
        start = back*2

        x_train = nn_x[start:]
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        y_train = nn_y[start:]
        return x_train,y_train

    def train_model(self, x_train, y_train):
        batch_size = 1
        model = self.create_model(batch_size=batch_size)
        model.fit(x_train, y_train,
                      epochs=100,
                      batch_size=batch_size, verbose=True)
        return model


    def create_model(self, batch_size = 8, weights = None):
        model = Sequential()
        model.add(LSTM(units=1, activation='linear',batch_input_shape=(batch_size, 7, 1),
                       input_dim=1, input_length=7, stateful=True, return_sequences=True))
        model.add(LSTM(units=1, activation='linear', stateful=False, return_sequences=False))
        model.add(Dropout(0.05))
        model.add(Dense(1, activation='relu'))
        model.summary()
        if weights is not None:
            model.set_weights(weights)
        model.compile(loss='mae',
                      optimizer='adam')
        return model

    def fit(self, series, npred = 28):
        batch_size = 1
        x_train, y_train = self.prepare_data(series)
        self.model = self.train_model(x_train, y_train)
        self.predict_start = series.index[-1] + pd.Timedelta(days=1)

        x_pred0 = x_train[-1, :, :]
        x_pred0 = np.roll(x_pred0, 1, axis=0)
        x_pred0[0, 0] = y_train[-1]
        self.xpred = x_pred0
        self.resid = series.copy(deep=True)

    def create_resid(self, series, x_train):
        result = self.model.predict(x_train).flatten()
        resid = pd.Series(result, index = series.index)
        return resid

    def predict(self, npred):
        preds = []
        xpred = self.xpred
        model = self.model
        for i in range(0, npred):
            reshaped = xpred.reshape(1, xpred.shape[0], xpred.shape[1])
            model_pred = model.predict(reshaped, batch_size=1)
            pred = model_pred.flatten()[0]
            preds.append(pred)
            xpred = np.roll(xpred, 1, axis=0)
            xpred[0, 0] = pred
        predict_end = self.predict_start + pd.Timedelta(days = npred-1)
        index = pd.date_range(self.predict_start, predict_end)
        return pd.Series(data= pred, index=index)



def main():
    np.seterr(all='raise')
    data = loader.load_product_class_data("rohliky.tsv", False)
    data = data[data.index < pd.Timestamp('2017-11-01')]
    series = data.groupby(pd.Grouper(freq='D'))['product_count'].sum().fillna(0)
    series = series.astype('float')
    n_preds = 28
    predictor = LSTM_Predictor()
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