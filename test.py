import numpy as np
import pandas as pd
from corsempy.model import Model as md
from corsempy.optimizer import Optimizer as opt
from corsempy.identifier import Identifier as id
from corsempy.stats import Statistics as stat


df1 = pd.read_csv('data_poli.csv')

mod = """xi_1~=x1+x2+x3
eta_1 ~= y1+y2+y3+y4
eta_2 ~= y5+y6+y7+y8
eta_1~ xi_1
eta_2~ eta_1 + xi_1"""

my_model = md(mod, df1)
print(my_model.load_data().columns)
