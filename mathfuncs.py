import numpy as np
from scipy.integrate import simps
from pyshtools.utils import Wigner3j as w3jsh

def minus1pow(m):
    ''' 
    Returns (-1)**m
    '''
    if m%2==0:
        return 1
    else:
        return -1

def w3j(l, s, l1, m, t, m1):
    '''
    Computes the wigner 3j symbol numerically.
    Inputs: l, s, l1, m, t, m1
    Output: wigner3j(l, s, l1, m, t, m1)
    '''
    dell = abs(l - l1)
    ind = s - dell
    return w3jsh(l1, l, t, m1, m)[0][ind]
