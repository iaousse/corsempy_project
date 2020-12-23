from enum import Enum


class SEMOperations(Enum):
    REGRESSION = '~'
    MEASUREMENT = '=~'
    COVARIANCE = '~~'
    TYPE = 'is'