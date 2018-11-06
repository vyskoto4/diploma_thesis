import pandas as pd
import configuration


def load_product_class_data(class_name, remove_2014=False):
    fname = +class_name
    data = pd.read_csv(fname, sep='\t',parse_dates=[2] )
    data.columns = ['product_id','product_name', 'timestamp', 'product_count', 'unit_coeff', 'price_per_unit','product_price', 'unit']
    data = data.set_index('timestamp').sort_index()
    if remove_2014:
        data = data[data.index>=pd.Timestamp('2015-01-01')]
    return data


def load_test_sku(sku, base_dir=None, start_of_period = None, end_of_period= pd.Timestamp('2018-01-01')):
    fname = base_dir +'SKU\\'+sku+".tsv"
    data = pd.read_csv(fname, sep='\t',parse_dates=[2] )
    data.columns = ['product_id','product_name', 'timestamp', 'product_count', 'unit_coeff', 'price_per_unit','product_price', 'unit']
    data = data.set_index('timestamp').sort_index()
    if start_of_period is not None:
        data = data[data.index>=start_of_period]
    data = data[data.index < end_of_period]
    series = data.groupby(pd.Grouper(freq='D'))['product_count'].sum().fillna(0)
    return series

def load_test_sku_price(sku_name,  base_dir=None, start_of_period = None, end_of_period=pd.Timestamp('2018-01-01')):
        fname = base_dir + 'SKU\\' + sku_name + ".tsv"
        data = pd.read_csv(fname, sep='\t', parse_dates=[2])
        data.columns = ['product_id', 'product_name', 'timestamp', 'product_count', 'unit_coeff', 'price_per_unit',
                        'product_price', 'unit']
        data = data.set_index('timestamp').sort_index()
        data = data[data.index < end_of_period]
        data = data[data.index >= start_of_period]
        series = data.groupby(pd.Grouper(freq='D'))['product_price'].mean().fillna(0)
        return series


def load_test_sku_prediction(sku, prediction_model, period, base_dir=None):
    resid =  configuration.FORECAST_RES_DIR +prediction_model+"\\" + sku +"\\"+str(period)+"_resid.pickle"
    forecast =  configuration.FORECAST_RES_DIR +prediction_model+"\\" + sku +"\\"+str(period)+"_result.pickle"
    s_resid  = pd.read_pickle(resid)
    s_forecast  = pd.read_pickle(forecast)
    return s_forecast, s_resid