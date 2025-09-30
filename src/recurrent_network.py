from typing import Callable
from recurrent_layer import *
from layer import *
from network import *

class RecurrentNetwork(Network):
    def __init__(self, layer_sizes:list[int], active:Callable, learn:Callable, random_seed:int=42):
        '''
        recurrent_index: index of the layer that should be recurrent (default = first hidden layer)
        '''
        self.layers = []

        # Starts iterating at index 1 to exclude input size as its own layer
        for s, size in enumerate(layer_sizes[1:], start=1):
            new_layer   = RecurrentLayer(
                num_neurons = size,
                num_inputs  = layer_sizes[s-1], # Preceeding output as input to the current layer
                active      = active,
                learn       = learn,
                random_seed = random_seed
            )
            self.layers.append(new_layer)

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

def main():
    # Setup:
    text = "The quick brown fox jumped over the lazy dog"

    # Get unique characters
    vocab = sorted(set(text.split()))
    vocab_size = len(vocab)

    # Maps
    word_to_idx = {c: i for i, c in enumerate(vocab)}
    idx_to_word = {i: c for i, c in enumerate(vocab)}

    # Encode Text as One-Hot Vectors
    def one_hot(word):
        vec = [0] * vocab_size
        vec[word_to_idx[word]] = 1
        return vec
    
    sequence = [one_hot(w) for w in text.split()]   # entire text as a sequence
    # Initialize small RNN
    # 3 input neurons (vocab size), 5 hidden neurons, 3 output neurons (vocab size)
    rnn = RecurrentNetwork([vocab_size, 16, vocab_size], active=Activation.relu, learn=LearningRule.gradient)

    def mse(y_true, y_pred, n):
        result = 0
        for i in range(len(y_true)):
            result += (y_true[i] - y_pred[i])**2

        return result/n
    
    # TRAINING
    for epoch in range(1000):
        total_loss = 0
        for i in range(len(sequence)-1):
            x = sequence[i]       # input char one-hot
            y_true = sequence[i+1] # next char one-hot

            rnn.train(x, y_true, 0.1, 0.9)  # you’d need a train method using BPTT
            y_pred  = rnn.predict(x)
            total_loss += mse(y_true, y_pred, len(y_true))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss {total_loss}")

   # Suppose your text and vocab setup are already done
    seed_words = ["the", "quick", "brown"]
    seed_sequence = [one_hot(w) for w in seed_words]

    # Predict next 5 words
    n_next = 5
    predicted_sequence = rnn.predict_next(seed_sequence, n_next)

    # Convert predictions back to words (take argmax)
    predicted_words = []
    for out in predicted_sequence[len(seed_sequence):]:  # only the new predictions
        idx = out.index(max(out))
        predicted_words.append(idx_to_word[idx])

    print("Predicted next words:", " ".join(predicted_words))

    return

if __name__ == "__main__":
    main()