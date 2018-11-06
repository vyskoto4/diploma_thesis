from collections import Counter
from gurobipy import *
import numpy as np

from inventory import extensive_form_optimization
import utils


def progressive_hedging(scenarios, b, c, h, f, ro0, durability = None, max_iter = 100):
    """

    :param scenarios: list of pairs of the demand scenarios,
                        pair[0] - list of demand values at T=0,1,2.. of the scenario,
                        pair[1] - scenario probability
    :param b: shortage costs at times 0...T
    :param c: ordering costs at times 0...T
    :param h: holding cost at times 0...T
    :param lambdas: lambda coefficients for PH
    :param ros: ro coefficients for PH
    :param fixed_lambdas: if True, lambda coefficients won't be update during PH execution
    :param fixed_ros: if True, ro coefficients won't be update during PH execution
    :param max_iter: maximum algo iterations
    :return:
    """

    scenario_structs = build_scenario_structs(scenarios=scenarios, b=b, c=c, h=h, f=f, durability=durability)
    clear_scenarios = []
    for s in scenarios:
        clear_scenarios.append(s[0]) # throw away prob - saved in structs now
    return hedge(scenario_structs,  ro = ro0, max_iter=max_iter)

def hedge(scenario_structs,  ro, gamma = 1.6, theta = 0.0001, max_iter = 30, max_min_dif_iter = 10, max_res_reps = 10):
    print("Will start first PH iteration...")
    xhats = None
    x_hats_diff_min = 1.5/scenario_structs[0].t
    x_hats_diff = np.inf
    x_hats_diff_counter = 0
    xhats_rep_count = Counter()
    x_hist = [set() for i in range(scenario_structs[0].t)]
    for i in range(0, max_iter):
        results = []
        for  scenario_ind in range(len(scenario_structs)):
            struct = scenario_structs[scenario_ind]
            #print("Solving scenario  {}".format(scenario_ind))
            if xhats is not None:
                subres = (struct.solve(xhats = xhats, ro=ro))
            else:
                subres = (struct.solve())
            results.append(subres)
        results = np.vstack(results)
        old_xhats,xhats = xhats,np.mean(results, axis = 0)
        rounded_xhats = np.round(xhats)
        xhat_hash ="_".join([ str(i) for i in rounded_xhats])
        xhats_rep_count[xhat_hash]+=1
        if old_xhats is not None:
            x_hats_diff = np.mean(np.abs(rounded_xhats - np.round(old_xhats)))
            if x_hats_diff > x_hats_diff_min:
                x_hats_diff_counter = 0
            else:
                x_hats_diff_counter+=1
            for xhat_ind in range(len(xhats)):
                xh = int(rounded_xhats[xhat_ind])
                if xh in x_hist[xhat_ind]:
                    xhats[xhat_ind] = int(np.round(max(results[:,xhat_ind])))
                    fix_x(scenario_structs, xhat_ind, xhats[xhat_ind])
                x_hist[xhat_ind].add(int(rounded_xhats[xhat_ind]))
        if check_termination(results, xhats, theta) or xhats_rep_count[xhat_hash] >= max_res_reps or  x_hats_diff_counter >= max_min_dif_iter :
            if x_hats_diff_counter >= max_min_dif_iter:
                print("PH met termination difference criterion, returning {}".format(xhats))
            elif xhats_rep_count[xhat_hash] >= max_res_reps:
                print("PH met termination solution repeat counter criterion, returning {}".format(xhats))
            else:
                print("PH met theta termination criterion, returning {}".format(xhats))
            break
        ro = gamma*ro
        print("PH iteration ",i, " done, xhats diff",x_hats_diff ,"result ", xhats)
    return xhats.tolist()

def fix_x(scenario_structs, xhat_ind, x_hat):
    print("Fixing x{} to {}".format(xhat_ind, x_hat))
    for scenario_struct in scenario_structs:
        scenario_struct.set_xhat(x_hat, xhat_ind)

def check_termination(results, xhats, theta):
    for i in range(0, len(xhats)):
        col = results[:,i]
        x_hat_col = (np.full(len(col), xhats[i]))
        save_x_hat_col = np.copy(x_hat_col)
        save_x_hat_col[x_hat_col == 0] = 1
        diff  = np.sum( (col - x_hat_col)**2)
        rounded = np.round(diff, decimals = 2)
        if rounded == 0:
            continue
        if xhats[i] == 0:
            return False
        if math.sqrt(diff)/xhats[i] > theta:
            return False
    return True

def mape(y_true, y_pred):
    save_div = y_true
    save_div[y_true==0] = 1
    return np.mean(np.abs((y_true - y_pred) / save_div)) * 100


def build_scenario_structs(scenarios, b, c, h, f, durability ):
    b = utils.to_list(b, len(scenarios[0][0]))
    c = utils.to_list(c, len(scenarios[0][0]))
    h = utils.to_list(h, len(scenarios[0][0]))
    f = None if f is None else utils.to_list(f, len(scenarios[0][0]))
    scenario_structs = []
    for scenario, prob in scenarios:
        scenario_structs.append(ScenarioStruct(scenario=scenario, prob=prob, b =b,c=c, h=h,f=f, durability=durability))
    return scenario_structs



class ScenarioStruct:
    def __init__(self, scenario, prob, b, c, h, f, durability, lambda0 = 0.01):
        self.scenario = scenario # demands at time 0...T
        self.prob = prob
        self.base_objective_sum = [] # sum of objective terms without the nonantipacitivity penalties
        self.t = len(scenario)
        self.penalty_terms =[]
        self.build_model(b=b,c=c,h=h, f=f, durability=durability)
        self.lambdas = [lambda0 for i in range(self.t)]

    def build_model(self, b, c, h, f, durability=None):
        self.tree = extensive_form_optimization.ScenarioTree([[self.scenario, 1]], b = b, c = c, h = h, f=f, durability = durability)
        self.tree.model.setParam('OutputFlag', False)
        #self.tree.model.write('file1.lp')

    def get_x(self, t):
        return self.tree.get_x_var(t)

    def get_x_value(self, t):
        return self.get_x(t).x

    def set_xhat(self,xhat, xhat_ind):
        variable = self.get_x(xhat_ind)
        self.tree.model.addConstr(variable == xhat)

    def create_penalty_terms(self, x_hats, ro):
        self.penalty_terms =[]
        assert(len(x_hats) == self.t)
        for t in range(0, len(x_hats)):
            x_hat = x_hats[t]
            lamb = self.lambdas[t]
            xt = self.get_x(t)
            self.penalty_terms.append(lamb * (xt - x_hat))
            self.penalty_terms.append(0.5 * ro * (xt - x_hat) * (xt - x_hat))

    def get_base_objective(self):
        return self.tree.obj_sum

    def solve(self, xhats = None, ro = None):
        if ro is not None and xhats is not None:
            self.update_lambdas(xhats, ro)
            self.create_penalty_terms(xhats, ro)
        objective_list = self.get_base_objective() + self.penalty_terms
        model = self.tree.model
        model.setObjective(quicksum(objective_list), GRB.MINIMIZE)
        model.optimize()
        return [self.get_x_value(i) for i in range(0,self.t)]

    def update_lambdas(self, x_hats , ro):
        for i in range(0, len(x_hats)):  # yes it should be -1, we do not make any decisions at the leaf nodes
            x_k = self.get_x_value(i)
            x_hat = x_hats[i]
            old_lamb = self.lambdas[i]
            self.lambdas[i] = old_lamb + ro * (x_k - x_hat)