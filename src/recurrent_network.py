from typing import Callable
from recurrent_layer import *
from layer import *
from network import *

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

    def forward_sequence(self, sequence:list[list[float]]) -> list[list[float]]:
        '''
        Predict outputs for a sequence of inputs.
        sequence: list of input vectors for each timestep
        returns: list of output vectors for each timestep
        '''

        output_sequence = []
        for vector in sequence:
            output  = vector
            for layer in self.layers:
                output  = layer.forward(output)
            output_sequence.append(output)
        return output_sequence
    
    def predict_next(self, seed_sequence:list[list[float]], n:int) -> list[list[float]]:
        '''
        Predict the next n outputs given the sequence
        seed_sequence: list of one-hot input vectors
        n: number of future words to predict
        '''
        # Initialize the hidden state with the seed sequence
        output_sequence = self.forward_sequence(seed_sequence)
        last_input      = seed_sequence[-1]

        for _ in range(n):
            # Predict the next word based on last output
            next_output = last_input
            for layer in self.layers:
                next_output = layer.forward(next_output)
            output_sequence.append(next_output)

            # The predicted output becomes the next input
            last_input  = next_output

        return output_sequence