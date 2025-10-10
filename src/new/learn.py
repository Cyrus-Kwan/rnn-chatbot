import numpy as np
from typing import Callable

from derivative import *

def hebbian(v:np.ndarray, w:np.ndarray, c:float, active:Callable):
    """
    Parameters:
        v: input vector
        w: weight vector
        c: learning constant
    """
    r   = active(v @ w)
    for i in range(len(w)):
        w[i]    += c * r * v[i]

    return

def discrete(v:np.ndarray, w:np.ndarray, d:float, c:float, active:Callable):
    """
    Parameters:
        v: input vector
        w: weight vector
        z: output vector
        u: derivative of activation function f'(x)
        d: desired output / target value
        c: learning constant
    """
    z   = active(v @ w)
    r   = d - z

    for i in range(len(w)):
        w[i]    += c * r * v[i]

    return

def SGD(v:np.ndarray, w:np.ndarray, d:float, c:float, du:str, active:Callable):
    """
    Parameters:
        v: input vector
        w: weight vector
        z: output vector
        u: derivative of activation function f'(x)
        d: desired output / target value
        c: learning constant
    """
    z   = active(v @ w)
    u   = dmap[du]
    e   = d - z

    r   = e * u(z)
    
    for i in range(len(w)):
        w[i]    += c * r * v[i]

    return

lmap   = {
    "hebbian"   :hebbian,
    "discrete"  :discrete,
    "sgd"       :SGD,
}