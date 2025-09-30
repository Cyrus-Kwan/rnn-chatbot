from typing import Callable
from neuron import *

class Layer():
    def __init__(self, num_neurons: int, num_inputs: int, active: Callable, learn: Callable):
        self.outputs    = [0.0] * num_neurons
        self.num_inputs = num_inputs
        self.learn      = learn
        self.derivative = None

        # Set the derivative function for backpropagation
        if active == Activation.bipolar:
            self.derivative = Derivative.bipolar
        elif active == Activation.logistic:
            self.derivative = Derivative.logistic
        
        # Initialize an array of zero'd neurons equal to the params
        self.neurons    = [
            Neuron(
                inputs  = [0.0] * num_inputs + [1], # Include bias term input
                weights = [0.0] * num_inputs + [1], # Include bias term weight
                active  = active,
                learn   = learn
            ) for n in range(num_neurons)
        ]

    def forward(self, inputs: list[float]) -> list[float]:
        '''
        The outputs of each layer are used as inputs for the proceeding layers.
        '''
        outputs = []
        for neuron in self.neurons:
            for v, vector in enumerate(inputs):
                neuron.inputs[v] = vector
            outputs.append(neuron.active(neuron))

        self.outputs    = outputs
        return outputs

def main():
    return

if __name__ == "__main__":
    main()