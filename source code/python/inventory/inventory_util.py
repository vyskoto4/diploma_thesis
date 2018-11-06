import numpy as np
import utils


def evaluate(real_scenario, xts, goods_info):
    t = len(real_scenario)
    b = utils.to_list(goods_info['b'],t)
    c = utils.to_list(goods_info['c'],t)
    h = utils.to_list(goods_info['h'],t)
    f = utils.to_list(goods_info['f'],t) if goods_info['f'] is not None else None
    durability = goods_info['durability']
    yt = 0
    fixed_cost = 0
    ordering_cost = np.sum(np.array(c)*(np.array(xts)))
    holding_cost = 0
    shortage_cost = 0
    leftovers = np.zeros(durability)
    if f is not None:
        fixed_cost = np.sum(np.array(f)*(np.array(xts)>0))
    for i in range(0,t):
        xt = xts[i]
        dt = real_scenario[i]
        holding_cost += h[i]*max(0,(xt+yt - dt ))
        shortage_cost += b[i] * max(0, (dt - xt - yt))
        if durability is not None and i >= durability:
            yt = max(xt+yt - dt - leftovers[-1],0)
        else:
            yt = max(xt + yt - dt, 0)
        if durability is not None:
            lt = dt
            leftovers = np.roll(leftovers, 1)
            for i in range(durability-1,0, -1):
                if lt == 0:
                    break
                dec = min(lt,leftovers[i])
                leftovers[i]-= dec
                lt-=dec
            leftovers[0] = 0
            leftovers[0] = min(xt,max(yt - np.sum(leftovers),0))
            yt = max(0, np.sum(leftovers))

    cost = fixed_cost+ordering_cost+holding_cost+shortage_cost
    return cost, fixed_cost, ordering_cost, holding_cost, shortage_cost
