import numpy as np

def tanh(x:float):
    """
    Params:
        x: singular float
    Returns:
        float
    """
    return np.tanh(x)

def sigmoid(x:float):
    """
    Params:
        x: singular float
    Returns:
        float
    """
    return 1/(1+np.exp(x))

def bipolar_sigmoid(x:float):
    """
    Params:
        x: singular float
    Returns:
        float
    """
    return (2/(1+np.exp(-x))) - 1

def relu(x:float):
    """
    Params:
        x: singular float
    Returns:
        float
    """
    return max(0, x)

def softmax(v:list[float]):
    """
    Params:
        v: vector of input values
    Returns:
        np.ndarray of probabilities that sum to 1
    """
    shift   = [x - max(v) for x in v]
    exps    = [np.exp(x) for x in shift]
    return np.array([x / sum(exps) for x in exps])

amap   = {
    "tanh"              : tanh,
    "sigmoid"           : sigmoid,
    "bipolar_sigmoid"   : bipolar_sigmoid,
    "relu"              : relu,
    "softmax"           : softmax
}