from typing import Callable
from layer import *

class RecurrentLayer(Layer):
    def __init__(self, num_neurons:int, num_inputs:int, active:Callable, learn:Callable, random_seed:int):
        '''
        Difference from FNN:
        1. Adds a hidden state vector for feedback (what the network has seen at previous timesteps)
            h(t) = f[W(x)*x(t) + W(h)*h(t-1) + b]
            h(t) -> hidden state (at current timestep t) or previous outputs
            W(x) -> input weights
            W(h) -> hidden weights (feedback)
            b    -> bias
            f    -> activation function
        2. Each forward pass uses current input + previous hidden state
        3. Stores the previous output for the next timestep
        '''
        super().__init__(num_neurons, num_inputs, active, learn, random_seed)

        self.hidden_state   = [0 for neuron in self.neurons]    # Output from previous input vector
        self.hidden_weights = [                                 # Weights fed back into neurons in this layer
            [
                random.uniform(-0.5, 0.5) for _ in range(num_neurons)
            ] for _ in range(num_neurons)
        ]
    
    def forward(self, inputs:list[float]) -> list[float]:
        outputs = []
        for n, neuron in enumerate(self.neurons):
            neuron.inputs[:-1]  = inputs        # Do not overwrite bias
            h_scale = sum(i * w for i, w in zip(self.hidden_state, self.hidden_weights[n]))
            scalar  = neuron.scalar() + h_scale
            outputs.append(self.active(scalar))

        self.hidden_state   = outputs

        return outputs