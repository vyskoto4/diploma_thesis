import numpy as np

from inventory import p_hedging_optimization, extensive_form_optimization


class InventoryModel:

    def get_orders(self, goods_info, scenarios, distribution):
        """
        :param goods_info: dict containing goods duration and costs - f,c, h, b
        :param scenarios:  scenarios - list of pairs - (scenario_chain, probability)
        :param distribution: dict of two lists - mus[], sigmas[]
        :return:  x_hats - the orders a times t...T
        """
        return None


class BaseStockModel(InventoryModel):
    def get_orders(self, goods_info, scenarios, distribution):
        mus = np.array(distribution['mus'])
        sigmas =  np.array(distribution['sigmas'])
        return (mus + 1.96*sigmas).tolist()


class HedgingModel(InventoryModel):
    def get_orders(self, goods_info, scenarios, distribution):
        b = goods_info['b']
        c = goods_info['c']
        h = goods_info['h']
        f = goods_info['f']
        durability = goods_info['durability']
        return p_hedging_optimization.progressive_hedging(scenarios, b, c, h, f, durability=durability, ro0=1.0)

class TimeLimitModel(InventoryModel):
    def get_orders(self, goods_info, scenarios, distribution):
        b = goods_info['b']
        c = goods_info['c']
        h = goods_info['h']
        f = goods_info['f']
        durability = goods_info['durability']
        return extensive_form_optimization.multi_period(scenarios, b, c, h, f, durability=durability, timelimit=10*60)


class ScenarioSubsetModel(InventoryModel):
    def get_orders(self, goods_info, scenarios, distribution, scenario_count = 25):
        b = goods_info['b']
        c = goods_info['c']
        h = goods_info['h']
        f = goods_info['f']
        my_scenarios = scenarios[0:scenario_count]
        durability = goods_info['durability']
        return extensive_form_optimization.multi_period(my_scenarios, b, c, h, f, durability=durability, timelimit=20*60)