import json
import os
import time

import numpy as np

import configuration
import loader
from inventory import inventory_util
from inventory.inventory_models import HedgingModel, ScenarioSubsetModel, BaseStockModel

NUM_SCENARIOS = 10
SELECTED_PREDICTOR = 'ARIMA'
ORDER_PRICE_MULTIPLIER = 0.65
HOLDING_MULTIPLIER = 1.0/50

models = {'BaseStock': BaseStockModel(),
          'Hedging': HedgingModel(),
          'Subset': ScenarioSubsetModel()}


RES_DIR = configuration.BASE_DIR+'\\results\\inventory\\'


def remove_if_needed(path):
    if os.path.isfile(path):
        os.remove(path)

def save_result(results):
    fname = RES_DIR+'results.json'
    remove_if_needed(fname)
    with open(fname, 'w') as outfile:
        json.dump(results, outfile,indent=4)

def init_result():
    fname = RES_DIR+'results.json'
    if os.path.isfile(fname):
        with open(fname) as f:
            results = json.load(f)
        return results

    results = {}
    for sku in configuration.SKUS:
        results[sku] = [{},{},{}]
    return results

def generate_scenarios(forecast,resid):
    prob = 1.0/NUM_SCENARIOS
    arr = np.zeros(shape=(NUM_SCENARIOS, configuration.N_PREDS))
    sigma = np.sqrt(resid.var())
    multiplier = 1.0167
    sigmas = []
    for i in range(configuration.N_PREDS):
        mu = forecast[i]
        arr[:,i] = [int(i) for i in np.random.normal(mu, sigma, NUM_SCENARIOS)]
        sigma*=multiplier
        sigmas.append(sigma)
    scenarios = []
    arr = arr.astype(int)
    for i in range(NUM_SCENARIOS):
        scenarios.append([arr[i,:],prob])
    distribution = {'mus': np.array(forecast.values), 'sigmas':np.array(sigmas)}
    return scenarios, distribution


def generate_goods_info(price_data, sku):
    ret = {}
    ret['b'] = price_data.values
    ret['c'] = price_data.values*ORDER_PRICE_MULTIPLIER
    ret['h'] = price_data.values[0]*HOLDING_MULTIPLIER
    ret['f'] = None
    ret['durability'] = configuration.SKU_DURABILITY[sku]
    return ret


def load_data(sku, period_index):
    period =  configuration.PERIODS[period_index]
    period_start = period[0]
    period_end = period[-1]
    prices = loader.load_test_sku_price(sku, base_dir=configuration.BASE_DIR, start_of_period=period_start, end_of_period=period_end)
    forecast, resid = loader.load_test_sku_prediction(sku, SELECTED_PREDICTOR, period_index, base_dir=configuration.BASE_DIR)
    observation = loader.load_test_sku(sku, base_dir=configuration.BASE_DIR, start_of_period=period_start, end_of_period=period_end)
    return prices, forecast, resid, observation

def base_benchmark():
    results = init_result()
    for model_name, model in models.items():
        for sku in configuration.SKUS:
            for period_ind in range(len(configuration.PERIODS)):
                print("Evaluating model {}, on sku {}, period {}".format(model_name, sku, str(period_ind)))
                prices, forecast, resid, observation = load_data(sku, period_ind)
                scenarios, distribution = generate_scenarios(forecast,resid)
                goods_info = generate_goods_info(prices, sku)
                start_time = time.time()
                orders = model.get_orders( goods_info, scenarios,distribution)
                cost, fixed_cost, ordering_cost, holding_cost, shortage_cost = inventory_util.evaluate(observation.values, orders, goods_info)
                execution_time = time.time() - start_time
                results[sku][period_ind][model_name] = {'orders': orders, 'cost':cost, 'fixed_cost': fixed_cost,
                                                   'ordering_cost':ordering_cost, 'holding_cost':holding_cost,
                                                   'shortage_cost':shortage_cost,
                                                    'time':execution_time}
                #print("Evaluating model {}, on sku {}, period {}, cost {}".format(model_name, sku, str(period_ind), cost))
                save_result(results)


if __name__ == '__main__':
    base_benchmark()