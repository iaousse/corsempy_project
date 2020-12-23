import numpy as np
import pandas as pd
from corsempy.model import Model as md
from corsempy.optimizer import Optimizer as opt
from corsempy.identifier import Identifier as id
from corsempy.stats import Statistics as stat
mtcars = pd.read_csv('mtcars.csv')
mod = """
mpg ~=mpg
cyl ~=cyl
disp ~=disp
disp ~=disp
hp ~=hp
qsec ~=qsec
wt ~=wt
mpg ~ cyl + disp + hp
qsec ~ disp + hp + wt"""
my_md = md(mod, mtcars)
print(my_md.structure())
