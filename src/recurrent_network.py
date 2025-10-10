from typing import Callable
from recurrent_layer import *
from layer import *
from network import *
import numpy as np

class RecurrentNetwork(Network):
    def __init__(self, layer_sizes:list[int], active:Callable, learn:Callable, loss:str="cross-entropy", random_seed:int=42):
        '''
        recurrent_index: index of the layer that should be recurrent (default = first hidden layer)
        '''
        super().__init__(layer_sizes, active, learn, loss, random_seed)

        self.layers = []

        # Starts iterating at index 1 to exclude input size as its own layer
        for s, size in enumerate(layer_sizes[1:-1], start=1):
            hidden_layer    = RecurrentLayer(
                num_neurons = size,
                num_inputs  = layer_sizes[s-1], # Preceeding output as input to the current layer
                active      = active,
                learn       = learn,
                random_seed = random_seed
            )
            self.layers.append(hidden_layer)

        if loss == "mse":
            output_layer    = Layer(
                num_neurons = layer_sizes[-1],
                num_inputs  = len(self.layers[-1].neurons),
                active      = active,
                learn       = learn,
                random_seed = random_seed
            )
        elif loss == "cross-entropy":
            # Output layer is regular feedforward layer
            output_layer    = Layer(
                num_neurons = layer_sizes[-1],
                num_inputs  = len(self.layers[-1].neurons),
                active      = Activation.softmax,
                learn       = learn,
                random_seed = random_seed
                )
        self.layers.append(output_layer)

    def train(self, inputs:list[list[float]], targets:list[float], learning_rate:float, momentum:float):
        '''
        Parameters:
            inputs: 2D array / array of vectors where each row is a input vector at time t
                    e.g., sentence --> words
            targets: 2D array where each row is the expected output vector and columns is the
                     expected single value for that output neuron
        '''
        

    def predict(self, inputs):
        return