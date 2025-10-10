import random
import numpy as np
from typing import Callable

from active import *
from learn import *
from derivative import *

class NeuralNetwork:
    def __init__(self, layer_sizes:list[int], actives:list[str]=None, learn:str="SGD", loss:str="mse", random_seed:int=42):
        """
        Parameters:
            layer_sizes: determines the shape of neurons and weights for each layer
            actives: activation functions for each layer in the the network
            random_seed: reproducible random seed
        """
        random.seed(random_seed)

        self.layers     = []
        self.derivative = None
        self.loss       = loss

        self.learn      = self._get_learn(learn)
        self.actives    = self._get_actives(
            actives     = actives, 
            n_layers    = len(layer_sizes[1:])
        )

        for s, size in enumerate(layer_sizes[1:], start=1): # exclude input layer (has no weights)
            n_neurons   = size                              # number of neurons per layer
            n_weights   = layer_sizes[s-1]                  # inputs to current layer = outputs of previous layer 
                                                            # (+1 accounts for bias)

            # Initialize random weights to all neurons for each layer
            new_layer   = np.random.uniform(
                low     = -0.5,
                high    = 0.5,
                size    = (n_neurons, n_weights+1)
            )
            self.layers.append(new_layer)

        # Stores previous updates per layer
        self.velocities = [np.zeros_like(layer) for layer in self.layers]

    def train(self, v:np.ndarray, d:np.ndarray, c:float, m:float):
        """
        Paramters
            v: input vector
            d: target vector
            c: learning rate
            m: momentum
        """
        inputs  = [np.concatenate((v, [1]))]        # input vector with bias term
        outputs = []

        # Forward pass through the network    
        for l, layer in enumerate(self.layers):
            z = self.forward(v=inputs[-1], layer=layer, active=self.actives[l])
            inputs.append(np.concatenate((z, [1]))) # bias term for each input vector
            outputs.append(z)                       # output vectors without bias term

        # Calculate initial deltas at output layer
        deltas      = [None] * len(self.layers)
        deltas[-1]  = (d - outputs[-1]) * self.derivative[-1](outputs[-1])

        # Calculate hidden layer deltas (propagate backward)
        for l in reversed(range(len(self.layers[:-1]))):
            curr_layer  = self.layers[l]
            next_layer  = self.layers[l+1]

            curr_output = outputs[l]
            next_output = outputs[l+1]

            curr_delta  = deltas[l]
            next_delta  = deltas[l+1]

            # propagate delta
            deltas[l]   = (next_layer[:,:-1].T @ next_delta) * self.derivative[l](outputs[l])
            # next_layer[:,:-1] excludes the bias weights

        # Update weights
        for l, layer in enumerate(self.layers):
            curr_input  = inputs[l]
            curr_delta  = deltas[l]

            gradient    = np.outer(curr_delta, curr_input)

            # Apply update with learning rate (c)
            self.layers[l] += c * gradient

        return

    def forward(self, v:np.ndarray[float], layer:list[np.ndarray], active:Callable):
        """
        Parameters:
            v: input vector with bias
        Return:
            output of the specified layer
        """
        return active(layer @ v)
    
    def predict(self, v:np.ndarray[float]):
        """
        Parameters:
            v: input vector with bias
        Return:
            output vector of the entire network
        """
        inputs  = [np.concatenate((v, [1]))]
        outputs = []
        for l, layer in enumerate(self.layers):
            z = self.forward(v=inputs[-1], layer=layer, active=self.actives[l])
            inputs.append(np.concatenate((z, [1])))
            outputs.append(z)
        
        return outputs[-1]


    def _get_actives(self, actives:list[str], n_layers):
        """
        Parameters:
            actives: list of activation functions for each layer in the network
            n_layers: number of layers in the network
        """
        if actives:
            self.actives, self.derivative   = [
                (amap[active], dmap[active]) for active in actives
            ]
        else:
            self.actives, self.derivative   = [
                (amap["sigmoid"], dmap["sigmoid"]) for l in range(n_layers)
            ]
        return self.actives

    def _get_learn(self, learn:str):
        """
        Parameters:
            learn: the learning rule for updating the weights in the network
        """
        key = learn.lower()
        return lmap[key]