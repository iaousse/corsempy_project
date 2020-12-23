import numpy as np
import pandas as pd
from corsempy.model import Model
from corsempy.optimizer import Optimizer


class Identifier:
    def __init__(self, md: Model, op: Optimizer):
        self.md = md
        self.op = op

    def is_identified(self):
        """
        :return: whether the model is /is not identified (not complete)
        """
        m = len(self.md.structure()[0]['observed'])
        t = len(self.op.get_params())
        mat = self.md.get_beta()
        if t > m*(m+1)/2:
            print('The model is under identified!! number of parameters is greater than the number of equations')
        elif np.allclose(mat, np.tril(mat)) and sum(np.diagonal(mat) != 0) == 0:
            print('the model is recursive. The matrix B is strictly lower triangular')
            if np.all(self.md.get_psi() == np.diag(np.diagonal(self.md.get_psi()))):
                print("the model is identified")
            else:
                print("bow_free method is not available")
        else:
            print("check your model identification")
        return None


if __name__ == '__main__':
    df1 = pd.read_csv('data_poli.csv')
    df1 = df1.iloc[:, 1:]
    mod = """xi_1~=x1+x2+x3
        eta_1 ~= y1+y2+y3+y4
        eta_2 ~= y5+y6+y7+y8
        eta_1~ xi_1
        eta_2~ eta_1 + xi_1"""
    my_model = Model(mod, df1)
    my_opt = Optimizer(my_model, .5)
    my_id = Identifier(my_model, my_opt)
    my_id.is_identified()
