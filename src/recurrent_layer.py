from typing import Callable
from layer import *

class RecurrentLayer(Layer):
    def __init__(self, num_neurons:int, num_inputs:int, active:Callable, learn:Callable, random_seed:int):
        '''
        Difference from FNN:
        1. Adds a hidden state vector for feedback (what the network has seen at previous timesteps)
            h(t) = f[W(x)*x(t) + W(h)*h(t-1) + b]
            W(x) -> input weights
            W(h) -> hidden weights (feedback)
            b    -> bias
            f    -> activation function
        2. Each forward pass uses current input + previous hidden state
        3. Stores the previous output for the next timestep
        '''
        super().__init__(num_neurons, num_inputs, active, learn, random_seed)

        # Rows: Current neuron receiving feedback
        # Cols: Previous neuron sending feedback
        self.hidden_weights = [[0.0] * num_neurons] * num_neurons

        # Store previous hidden outputs
        self.prev_output    = [0.0] * num_neurons
    
    def forward(self, inputs:list[float]) -> list[float]:
        outputs = []
        for n, neuron in enumerate(self.neurons):
            # Copy input vector
            neuron.inputs[:-1]  = inputs    # Do not overwrite bias

            # Hidden state contribution: Each neuron receives a weighted sum of the previous outputs
            hidden_sum          = sum(h * w for h, w in zip(self.prev_output, self.hidden_weights[n]))

            # Add hidden contribution to bias term
            neuron.inputs[-1]   += hidden_sum   # bias + hidden feedback

            outputs.append(neuron.active(neuron))

        self.prev_output    = outputs
        self.outputs        = outputs
        return outputs