from typing import Callable
from layer import *

class Network():
    def __init__(self, layer_sizes:list[int], active:Callable, learn:Callable):
        '''
        Constructor creates layers in the network each with the specified activaiton function
        and learning signal.

        layer_sizes: input layer inclusive so a parameter of [3, 2, 1] would produce a network
        with 3 inputs, 2 neurons in the hidden layer, and one neuron in the output layer
        '''
        self.layers     = []
        
        # Starts iterating at index 1 to exclude input size as its own layer
        for s, size in enumerate(iterable=layer_sizes[1:], start=1):
            new_layer   = Layer(
                num_neurons = size, 
                num_inputs  = layer_sizes[s-1], # Preceeding output as input to the current layer
                active      = active,
                learn       = learn
            )

            self.layers.append(new_layer)

    def train(self, inputs:list[float], targets:list[float], learning_rate:float, momentum:float):
        '''
        Updates the weights of each neuron iterating over a sequence of input
        vectors and corresponding output vector labels.

        Performed in # steps:
            1. Forward pass performs the activation function on each neuron for the layers.
            The outputs of a layer are then used as inputs for each consecutive layer.
            2. Error calculation compares the output from each layer (in reverse order) to the
            target list. 
            3. Back propagation: Calculate the error signal Sk and Sj (for each layer)
                - Sk: between output and hidden layer
                - Sj: between hidden layer and input layer
            4. Calculate the deltas at each layer
            5. Update the weights with the deltas
        '''

        # 1. Forward pass
        activations = [inputs]  # Output from activation function
        scalars     = []        # Dot products of inputs and weights
        deltas      = []        # Weight updates for single training cycle

        # Compute the outputs for each layer
        for layer in self.layers:
            output  = layer.forward(activations[-1])
            scalar  = [neuron.scalar() for neuron in layer.neurons]

            activations.append(output)
            scalars.append(scalar)

        # 2. Error calculation: calculate the signal error (d-z) * f'(activation) for output layer
        #                       2D array because each array will hold outputs for each layer
        errors      = [[d - z for d, z in zip(targets, activations[-1])]]
        derivatives = [[self.layers[-1].derivative(z) for z in activations[-1]]]
        signals     = [[error * derivative for error, derivative in zip(errors[-1], derivatives[-1])]]

        # 3. Back propagation: Calculate the the error signal for each hidden layer
        #                      dotprod(output weights, output signal) * hidden derivative
        next_signal = signals[-1]
        next_layer  = self.layers[-1]
        for l, layer in enumerate(self.layers[-2::-1]):    # Exclude output layer
            curr_signal     = []        # 2D array signal for each neuron * layer
            curr_layer      = layer
            for o, output in enumerate(curr_layer.outputs):
                # Each current layer neuron corresponds to a weight of the next layer neurons
                neuron_signal   = []    # Temporary storage for dot product calculation
                for n, neuron in enumerate(next_layer.neurons):
                    neuron_signal.append(next_signal[n] * neuron.weights[o])
                curr_signal.append(sum(neuron_signal) * curr_layer.derivative(output))
            signals.append(curr_signal)

        # 4. Calculate deltas: Change for each weight in each neuron at each layer is equal to
        #                      learning rate * next signal * current output
        for layer, signal in zip(self.layers[::-1], signals):
            layer_deltas    = []
            for n, neuron in enumerate(layer.neurons):
                neuron_deltas   = []
                for w, weight in enumerate(neuron.weights):
                    # current layer outputs = next layer inputs
                    delta   = learning_rate * signal[n] * neuron.inputs[w]
                    neuron_deltas.append(delta)
                layer_deltas.append(neuron_deltas)
            deltas.append(layer_deltas)

        # 5. Update weights
        for l, layer in enumerate(self.layers[::-1]):
            for n, neuron in enumerate(layer.neurons):
                for w, weight in enumerate(neuron.weights):
                    delta                   = deltas[l][n][w]
                    neuron.weights[w]       += delta + momentum * neuron.prev_delta[w]
                    neuron.prev_delta[w]    = delta
        return
    
    def predict(self, inputs:list[float]):
        '''
        Expects an input vector depending on the number of input neurons and
        returns an output vector determined by the number of output neurons.
        '''
        for layer in self.layers:
            inputs  = layer.forward(inputs)

        return inputs

def main():
    return

if __name__ == "__main__":
    main()