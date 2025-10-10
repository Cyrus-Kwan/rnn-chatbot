from typing import Callable
from layer import *

class Network():
    def __init__(self, layer_sizes:list[int], active:Callable, learn:Callable, loss:str="mse", random_seed=42):
        '''
        Constructor creates layers in the network each with the specified activaiton function
        and learning signal.

        layer_sizes: input layer inclusive so a parameter of [3, 2, 1] would produce a network
        with 3 inputs, 2 neurons in the hidden layer, and one neuron in the output layer
        '''
        self.loss       = loss
        self.layers     = []
        
        # Starts iterating at index 1 to exclude input size as its own layer
        for s, size in enumerate(iterable=layer_sizes[1:], start=1):
            new_layer   = Layer(
                num_neurons = size, 
                num_inputs  = layer_sizes[s-1], # Preceeding output as input to the current layer
                active      = active,
                learn       = learn,
                random_seed = random_seed
            )

            self.layers.append(new_layer)

    def train(self, inputs:list[list[float]], targets:list[float], learning_rate:float, momentum:float):
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

        # Compute the outputs for each layer
        for layer in self.layers:
            output  = layer.forward(activations[-1])
            scalar  = [neuron.scalar() for neuron in layer.neurons]

            activations.append(output)
            scalars.append(scalar)

        # 2. Error calculation: 2D array because each array will hold outputs for each layer
        signals = [self._calculate_error(activations[-1], targets, self.layers[-1])]

        # 3. Back propagation: Calculate the the error signal for each hidden layer
        signals = self._backpropagation(signals)

        # 4. Calculate deltas: Change for each weight in each neuron at each layer is equal to
        deltas  = self._get_deltas(signals, learning_rate)

        # 5. Update weights
        self._update(deltas, signals, learning_rate, momentum)
        return
    
    def _calculate_error(self, output, targets, layer):
        signals = None
        if self.loss == "mse":
            # Calculate the initial signal error (d-z) * f'(activation) from output layer
            errors      = [d - z for d, z in zip(targets, output)]
            derivatives = [layer.derivative(z) for z in output]
            signals     = [error * derivative for error, derivative in zip(errors, derivatives)]
        elif self.loss == "cross-entropy":
            # Calculate the initial signal error (z-d) from output layer
            signals     = [d - z for d, z in zip(targets, output)]

        return signals

    def _backpropagation(self, signals:list[list[float]]):
        '''
        Calculates the error signal for each hidden layer
        The error signal for each neuron: dotprod(cw, cz) * f'(nz)

        Where:
            cw      : current neuron weights
            cz      : current neuron activation
            f'(x)   : derivative of activation function
            nz      : next neuron activation

        Params:
            signals : 2D array of error signals from output layer
        '''
        for l in reversed(range(len(self.layers[:-1]))):    # iterate in reverse and exclude output layer
            curr_signal = []
            next_signal = signals[-1]

            curr_layer  = self.layers[l]
            next_layer  = self.layers[l+1]

            for cn, curr_neuron in enumerate(curr_layer.neurons):
                # Decide whether to use pre-activation or post-activation for derivative
                if curr_layer.derivative == Derivative.relu:
                    z   = curr_neuron.scalar()                      # pre-activation
                else:
                    z   = curr_neuron.active(
                        curr_neuron.scalar(), curr_layer.neurons    # post-activation
                    )

                # Each current layer neuron corresponds to a weight of the next layer neurons
                neuron_signal   = []    # Temporary storage for dot product calculation
                for nn, next_neuron in enumerate(next_layer.neurons):
                    neuron_signal.append(next_signal[nn] * next_neuron.weights[cn])
                
                curr_signal.append(sum(neuron_signal) * curr_layer.derivative(z))
                
            signals.append(curr_signal)

        return signals
    
    def _get_deltas(self, signals:list[list[float]], learning_rate:float):
        '''
        Change for each weight in each neuron at each layer is equal to 
        Delta for each neuron: learning rate * next signal * current output

        Where:
            lr  : learning rate
            ns  : next neuron error signal
            cs  : current neuron activation
        '''
        deltas  = []    # Weight updates for single training cycle

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

        return deltas
    
    def _update(self, deltas:list[list[float]], signals:list[float], learning_rate:float, momentum:float):
        '''
        Updates the weight of each neuron by adding the delta to the current weight
        '''
        for l, layer in enumerate(self.layers[::-1]):
            for n, neuron in enumerate(layer.neurons):
                for w, weight in enumerate(neuron.weights):
                    delta                   = deltas[l][n][w]
                    neuron.weights[w]       += delta + momentum * neuron.prev_delta[w]
                    neuron.prev_delta[w]    = delta

    def predict(self, inputs:list[float]):
        '''
        Expects an input vector depending on the number of input neurons and
        returns an output vector determined by the number of output neurons.
        '''
        for layer in self.layers:
            inputs  = layer.forward(inputs)

        return inputs