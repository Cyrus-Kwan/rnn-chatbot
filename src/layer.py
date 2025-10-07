from typing import Callable
from neuron import *
import random

class Layer():
    def __init__(self, num_neurons:int, num_inputs:int, active:Callable, learn:Callable, random_seed:int):
        random.seed(random_seed)

        self.outputs    = [0.0 for _ in range(num_neurons)]
        self.num_inputs = num_inputs
        self.learn      = learn
        self.active     = active
        self.derivative = None

        # Set the derivative function for backpropagation
        if active == Activation.sign_zero:
            self.derivative = Derivative.sign_zero
        elif active == Activation.sign_one:
            self.derivative = Derivative.sign_one
        elif active == Activation.bipolar:
            self.derivative = Derivative.bipolar
        elif active == Activation.logistic:
            self.derivative = Derivative.logistic
        elif active == Activation.hyper_log:
            self.derivative = Derivative.hyper_log
        elif active == Activation.relu:
            self.derivative = Derivative.relu
        elif active == Activation.softmax:
            self.derivative = Derivative.softmax
        
        # Initialize an array of zero'd neurons equal to the params
        self.neurons    = [
            Neuron(
                inputs  = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)] + [1], # Include bias term input
                weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)] + [1], # Include bias term weight
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
            neuron.inputs[:-1]  = inputs    # Do not overwrite bias
            if self.active == Activation.softmax:
                outputs.append(neuron.active(self.neurons))
            else:
                outputs.append(neuron.active())

        self.outputs    = outputs
        return outputs