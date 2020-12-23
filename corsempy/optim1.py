import pandas as pd
import numpy as np
from corsempy.model import Model
from scipy.optimize import minimize


class Optimizer:
    """
    The optimizer class gets object of class Model and an arbitrary starting point
    """
    def __init__(self, md: Model):
        self.md = md

    def loss_func(self, params, loss_method='uls', compute_method='fim'):
        """

        :param params: a list of model parameters
        :param loss_method: 'uls', 'gls', 'ml'
        :param compute_method: 'jor', 'fim', 'new_fim1', 'new_fim2'
        :return: the loss function, distance between empirical covariance matrix and implied covarince matrix
        """
        if loss_method == 'uls':
            return md.u_least_squares(params, compute_method)
        elif loss_method == 'gls':
            return md.g_least_squares(params, compute_method)
        elif loss_method == 'ml':
            return md.max_likelihood(params, compute_method)
        else:
            print("error in loss_method")

    def fit_model(self, params, algo_method='BFGS', loss_method='uls', compute_method='fim'):
        """
        :param params: a list of model parametrs
        :param algo_method: algorithm of minimization
        :param loss_method: the descrpency function to use : 'uls', 'gls', 'ml'
        :param compute_method: 'jor', 'fim', 'new_fim1', 'new_fim2'
        :return:  a list of model parameters that minimizes the the loss_function
        """
        # results = minimize(self.loss_func, params, args=(loss_method, compute_method),
        # method=algo_method,
        # jac=None,
        # hess=None,
        # hessp=None,
        # bounds=None,
        # constraints={},
        # tol=None,
        # callback=None,
        # options={'maxiter': 1e3, 'ftol': 1e-8})
        results = minimize(self.loss_func, params, args=(loss_method, compute_method), method=algo_method, jac=None,
                           hess=None, hessp=None, tol=None, callback=None,
                           options={'disp': True})
        return results.x

if __name__ == '__main__':
    df1 = pd.read_csv('data_poli.csv')
    mod = """xi_1~=x1+x2+x3
    eta_1 ~= y1+y2+y3+y4
    eta_2 ~= y5+y6+y7+y8
    eta_1~ xi_1
    eta_2~ eta_1 + xi_1"""

