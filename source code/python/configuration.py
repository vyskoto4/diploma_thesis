import pandas as pd


BASE_DIR = 'C:\\Users\\tomas\\Desktop\\diplomka\\rohlik_dp\\'
FORECAST_RES_DIR = BASE_DIR+'results\\forecasting\\'

SKUS = ['SKU1','SKU2', 'SKU3','SKU4', 'SKU5']
SKU_DURABILITY = {'SKU1':4, 'SKU2':6, 'SKU3':8, 'SKU4':None, 'SKU5':None}

N_PREDS = 28

PERIODS = [(pd.Timestamp('2017-08-01'),pd.Timestamp('2017-08-29 00:00:00'))]
           #(pd.Timestamp('2017-09-01'),pd.Timestamp('2017-09-29 00:00:00')),
           #(pd.Timestamp('2017-10-01'),pd.Timestamp('2017-10-29 00:00:00'))]
