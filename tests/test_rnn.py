import unittest
import json

from pathlib import Path
from test_setup import *
from recurrent_network import *

def main():
    # Setup:
    cur_path:Path   = Path(__file__).parent.parent.resolve()
    print(cur_path)
    with open(cur_path / "data/green-eggs-and-ham.txt") as f:
        text = f.read()

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
    seed_words = ["The", "mouse", "is", "in", "the", "house."]
    seed_sequence = [one_hot(w) for w in seed_words]

    # Predict next 10 words
    n_next = 10
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