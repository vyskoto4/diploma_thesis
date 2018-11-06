from collections import defaultdict, Set
from gurobipy import *
import utils


# def single_stage(pdf, b, c, h ):
#     """
#
#     :param pdf: dict of the distribution, keys - demand value, values - demand probability
#     :param b:
#     :param c:
#     :param h:
#     :return:
#     """
#     m = Model()
#     x = m.addVar(name = 'x')
#     m.addConstr(x >= 0)
#     obj_sum = []
#     demands = list(pdf.keys())
#     for i in range(len(demands)):
#         demand = demands[i] #dk
#         prob = pdf[demand] # pk
#         tk = m.addVar(name = 't{}'.format(i))
#         obj_sum.append(tk*prob)
#         m.addConstr(tk >= (c-b)*x + b*demand)
#         m.addConstr(tk >= (c+h)*x - h*demand)
#     m.setObjective(quicksum(obj_sum), GRB.MINIMIZE)
#     m.optimize()
#     return m

def single_stage(pdf, b, c, h ):
    demands = list(pdf.keys())
    scenarios = []
    for i in range(len(demands)):
        demand = demands[i] #dk
        prob = pdf[demand] # pk
        scenarios.append(([demand], prob))
    model = multi_period(scenarios, b, c, h)
    return model


def multi_period(scenarios, b, c, h, f=None, durability = None, timelimit = None):
    """

    :param scenarios: list of pairs of the demand scenarios,
                        pair[0] - list of demand values at T=0,1,2.. of the scenario,
                        pair[1] - scenario probability
    :param b: shortage costs at times 0...T
    :param c: ordering costs at times 0...T
    :param h: stprage costs at times 0...T
    :param f: fixed ordering costs at times 0...T
    :return:
    """

    tree = ScenarioTree(scenarios, b = b, c=c, h =h, f=f, durability=durability)
    model = tree.model
    # if hedging_info is not None:
    #     tree.add_penalty_terms(x_hat_t= hedging_info.x_hats, lambdas=hedging_info.lambdas, ros = hedging_info.ros)
    model.setObjective(quicksum(tree.obj_sum), GRB.MINIMIZE)
    #model.write('file.lp')
    if timelimit is not None:
        model.setParam('TimeLimit', timelimit)
    model.optimize()
    res = [max(0,tree.get_x_var(t).x) for t in range(0, tree.t)]
    return res


class ScenarioTree:
    def __init__(self, scenarios, b, c, h,f=None,  m = None, durability=None):
        self.root = defaultdict(ScenarioTree.Node)
        self.model = Model()
        self.variables = {}
        self.obj_sum = []
        self.t = len(scenarios[0][0])
        self.c = utils.to_list(c,self.t)
        self.b = utils.to_list(b,self.t)
        self.h = utils.to_list(h,self.t)
        self.f = None if f is None else utils.to_list(f, self.t)
        self.M = self.find_m(scenarios, durability) if m is None else m
        self.durability = durability
        self.start_build_tree(scenarios)
        self.start_build_model()

    def find_m(self,scenarios, durability):
        max_m = 0
        max_durability = durability if durability is not None and durability>0 else len(scenarios[0][0])
        for s in range(0, len(scenarios)):
            for demand in scenarios[s][0]:
                max_m = max(max_m, demand*max_durability)
        return max_m

    def start_build_tree(self, scenarios):
        for s in range(0, len(scenarios)):
            scenario = scenarios[s][0]
            prob = scenarios[s][1]
            succ = self.root[scenario[0]]
            self.buildtree(succ, scenario, prob, 0)


    def buildtree(self, node, scenario, prob, t):
        node.prob = node.prob + prob
        node.demand = scenario[t]
        node.scenario = scenario[:t+1]
        if t < len(scenario) -1:
            succ = node.succ[scenario[t+1]]
            self.buildtree(succ, scenario, prob, t+1)

    def start_build_model(self):
        for i in range(0, self.t):
            xt = self.add_var('x', k='all', t=i,  vtype=GRB.INTEGER)
            self.model.addConstr(xt >= 0)
            if self.f is not None and self.f[i]>0:
                zt = self.add_var('z', k='all', t=i, vtype=GRB.BINARY)
                self.model.addConstr(xt <= self.M*zt)
                self.obj_sum.append(self.f[i]*zt)


        for succ in self.root.values():
            self.do_build_model(succ, 0, None)

    def do_build_model(self, node, t, y_k_t):
        x_t = self.get_x_var(t)
        k =  self.get_k_tag(node)# use demand value as k tag
        #k = '_'.join(map(str, node.scenarios)) # use demand value as k tag
        t_k_t = self.add_var('t', k = k, t = t, vtype= GRB.CONTINUOUS )
        demand = node.demand
        prob = node.prob
        c = self.c[t]
        b = self.b[t]
        h = self.h[t]
        if y_k_t is not None:
            # we are at T>0
            self.model.addConstr(t_k_t >= (c - b) * x_t + b * demand - b * y_k_t)
            self.model.addConstr(t_k_t >= (c + h) * x_t - h * demand + h * y_k_t)
        else:
            # we are at T=0,
            self.model.addConstr(t_k_t >= (c - b) * x_t + b * demand)
            self.model.addConstr(t_k_t >= (c + h) * x_t - h * demand)

        self.obj_sum.append(prob*t_k_t)

        if(t < self.t -1):
            if self.durability is not None and self.durability > 0 and t >= self.durability:
                expired = self.get_l(t-self.durability, t-1, self.get_k_pred(k, 1))
                expr = (x_t + y_k_t - expired -demand )
            elif y_k_t is not None:
                expr = (x_t + y_k_t - demand )
            else:
                expr = (x_t - demand)

            y_k_t_plus = self.add_var('y', k, t+1)
            if self.durability is not None and self.durability>0:
                leftovers = []
                for i in range(t - self.durability+1, t+1):
                    if i >= 0:
                        ltt = self.add_l(i, t,k, x_t)
                        leftovers.append(ltt)
                sumexpr = quicksum(leftovers)
                self.model.addConstr(y_k_t_plus==sumexpr)

            b_k_t = self.add_var('b', k, t+1, vtype= GRB.BINARY)
            self.model.addConstr(y_k_t_plus >= 0)
            self.model.addConstr(y_k_t_plus >= expr)
            self.model.addConstr(y_k_t_plus <= expr + self.M*b_k_t )
            self.model.addConstr(y_k_t_plus <= self.M*(1-b_k_t))
            for succ in node.succ.values():
                self.do_build_model(succ, t+1, y_k_t_plus)

    def get_l(self, order_time, cur_time, k):
        name = 'lk{}__{}_{}'.format(k,order_time, cur_time)
        return self.variables[name]

    def add_l(self, order_time, cur_time, k, xk):
        name = 'lk{}__{}_{}'.format(k,order_time, cur_time)
        l = self.model.addVar(name=name, vtype=GRB.INTEGER)
        if cur_time == order_time:
            self.model.addConstr( l <= xk)
        if cur_time > order_time:
            k_pred = self.get_k_pred(k)
            pred_name = 'lk{}__{}_{}'.format(k_pred,order_time, cur_time-1)
            pred = self.variables[pred_name]
            self.model.addConstr( l <=pred)
        self.variables[name] = l
        return l

    def get_k_pred(self, k, step =1):
        for i in range(len(k)-1,0, -1):
            if k[i] == '_':
                step-=1
                if step == 0:
                    return k[:i]
        return k

    # def add_penalty_terms(self, x_hat_t, lambdas, ros ):
    #     root_vals = list(self.root.values())
    #     assert len(root_vals) ==1
    #     #for use only with tree made of single scenario => PROGRESSIVE HEDGING
    #     node = root_vals[0]
    #     k = '_'.join(map(str, node.scenario))
    #     x_t = self.get_x0()
    #     for t in range(len(x_hat_t)):
    #         x_hat = x_hat_t[t]
    #         lamb = lambdas[t]
    #         ro = ros[t]
    #         self.obj_sum.append(lamb* (x_t - x_hat))
    #         self.obj_sum.append(0.5*ro* (x_t - x_hat)*(x_t - x_hat))
    #         if t== len(x_hat_t) -1: break
    #         k = self.get_k_tag(node)
    #         x_t = self.get_var('x', k=k, t=t+1)
    #         node = list(node.succ.values())[0]



    def get_k_tag(self, node):
        return '_'.join(map(str, node.scenario))

    class Node:
        def __init__(self):
            self.succ = defaultdict(ScenarioTree.Node)
            self.demands = 0 # mapping k -> demand scenario value
            self.scenario = []
            self.leaf = False
            self.prob = 0

    def add_var(self, base_name, k, t, vtype = None):
        v_name = '{}_t{}_k_{}'.format(base_name, t,k)
        if vtype is not None:
            var = self.model.addVar(name = v_name, vtype= vtype)
        else:
            var = self.model.addVar(name = v_name, vtype= GRB.INTEGER)
        self.variables[v_name] = var
        return var

    def get_var(self, base_name, k, t):
        v_name = '{}_t{}_k_{}'.format(base_name, t,k)
        return self.variables[v_name]

    def get_x_var(self, t):
        return self.get_var('x', t= t, k='all')



# def mult_step_policy(scenarios, b, c, h ):
#     """
#     :param pdf: dict of the distribution, keys - demand value, values - demand probability
#     :param b:
#     :param c:
#     :param h:
#     :return:
#     """
#     m = Model()
#     timesteps = len(scenarios[0][0])
#     vars = {}
#     for i in range(timesteps):
#         name ='x{}'.format(i)
#         x = m.addVar(name=name, vtype=GRB.INTEGER)
#         vars[name] = x
#         m.addConstr(x >= 0)
#     M = 0
#     for scenario_pair in scenarios:
#         M = max(M, sum(scenario_pair[0]))
#     scenario_ind = -1
#     obj_sum = []
#     for scenario_pair in scenarios:
#         scenario = scenario_pair[0]
#         prob = scenario_pair[1]
#         scenario_ind+=1
#         for timestep in range(timesteps):
#             t = m.addVar(name = 't{}__{}'.format(scenario_ind, timestep), vtype=GRB.INTEGER)
#             x = vars['x{}'.format(timestep)]
#             demand = scenario[timestep]
#             if(timestep > 0):
#                 x_prev = vars['x{}'.format(timestep-1)]
#                 yt_name='y{}__{}'.format(scenario_ind, timestep)
#                 yt = m.addVar(name=yt_name, vtype = GRB.INTEGER)
#                 vars[yt_name] = yt
#                 m.addConstr(yt >= 0)
#                 d_prev = scenario[timestep-1]
#                 if(timestep > 1):
#                     ytp_name = 'y{}__{}'.format(scenario_ind, timestep-1)
#                     expr = x_prev + vars[ytp_name] - d_prev
#                 else:
#                     expr = x_prev - d_prev
#                 m.addConstr(yt >= expr)
#                 b_k_t = m.addVar(name='b{}__{}'.format(scenario_ind,timestep), vtype=GRB.BINARY)
#                 m.addConstr(yt <= expr + M * b_k_t)
#                 m.addConstr(yt <= M * (1 - b_k_t))
#             else:
#                 yt = 0
#             m.addConstr(t >= (c[timestep] - b[timestep]) * x + b[timestep] * demand - b[timestep]*yt)
#             m.addConstr(t >= (c[timestep]+h)*x - h*demand + h*yt)
#             obj_sum.append(t*prob)
#     m.setObjective(quicksum(obj_sum), GRB.MINIMIZE)
#     m.optimize()
#     for i in range(timesteps):
#         name ='x{}'.format(i)
#         print(vars[name].X)
#     return m