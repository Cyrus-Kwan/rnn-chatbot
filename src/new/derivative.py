import numpy as np
from active import *

def _tanh(z:float):
    return 1 - z**2

def _sigmoid(z:float):
    return z * (1 - z)

def _bipolar_sigmoid(z:float):
    return 0.5 * (1 - z**2)

def _relu(z:float):
    if z > 1: return 1
    else    : return 0

def _softmax(z:float):
    s = softmax(z)
    s = s.reshape(-1, 1)  # column vector
    return np.diagflat(s) - np.dot(s, s.T)

dmap   = {
    "tanh"              : _tanh,
    "sigmoid"           : _sigmoid,
    "bipolar_sigmoid"   : _bipolar_sigmoid,
    "relu"              : _relu,
    "softmax"           : _softmax
}