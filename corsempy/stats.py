import numpy as np
from corsempy.model import Model
from corsempy.optimizer import Optimizer


class Statistics:
    def __init__(self, md: Model, op: Optimizer):
        self.md = md
        self.op = op

    def degrees_of_freedom(self, compute_method='jor'):
        num_observed = len(self.md.structure()[0]['observed'])
        num_para = self.op.get_params().size()
        if compute_method == 'jor' or compute_method == 'new_fim2':
            return (num_observed*(num_observed+1))/1 - num_para
        else:
            print('currently available just for jor and new_fim2')
            return None

    def compute_gfi(self, compute_method='jor'):
        sample_cov = self.md.load_data()
        res_para = self.op.fit_model(self.op.get_params())
        if compute_method == 'jor':
            implied_sigma = self.op.compute_sigma_jor(res_para)
            resid_matrix = sample_cov - implied_sigma
            return np.trace(resid_matrix.dot(resid_matrix.T)) / (sample_cov ** 2).sum()
        elif compute_method == 'fim':
            implied_sigma = self.op.compute_sigma_fim(res_para)
            resid_matrix = sample_cov - implied_sigma
            return np.trace(resid_matrix.dot(resid_matrix.T)) / (sample_cov ** 2).sum()
        elif compute_method == 'new_fim1':
            implied_sigma = self.op.compute_sigma_new_fim1(res_para)
            resid_matrix = sample_cov - implied_sigma
            return np.trace(resid_matrix.dot(resid_matrix.T)) / (sample_cov ** 2).sum()
        elif compute_method == 'new_fim2':
            implied_sigma = self.op.compute_sigma_new_fim2(res_para)
            resid_matrix = sample_cov - implied_sigma
            return np.trace(resid_matrix.dot(resid_matrix.T)) / (sample_cov ** 2).sum()
        else:
            print('problem in compute method!')
            return None

    def compute_agfi(self, compute_method='jor'):
        m = len(self.md.structure()[0]['observed'])
        if compute_method == 'jor' or compute_method == 'new_fim2':
            return 1-(m*(m+1)/(2*self.degrees_of_freedom(compute_method)))*(1-self.compute_gfi(compute_method))
        else:
            print('not currently available')
            return None
    # def compute_chi2(self):


if __name__ == '__main__':
    print('ok')
