import unittest
import json

from pathlib import Path
from test_setup import *
from neuron import *
from layer import *

class TestLayer(unittest.TestCase):
    # helper for comparing float sequences
    def assertSequenceAlmostEqual(self, first, second, places=3):
        self.assertEqual(len(first), len(second), "Lengths differ")
        for i, (a, b) in enumerate(zip(first, second)):
            with self.subTest(index=i):
                self.assertAlmostEqual(a, b, places=places)

    def setUp(self):
        cur_path:Path   = Path(__file__).parent.resolve()
        with open(cur_path / "test_layer.json") as f:
            self.test_cases:dict    = json.load(f)
            self.active:dict        = {
                "sign_zero":Activation.sign_zero,
                "sign_one":Activation.sign_one,
                "logistic":Activation.logistic,
                "bipolar":Activation.bipolar,
                "hyper_log":Activation.hyper_log
            }
            self.learn:dict         = {
                "hebbian":LearningRule.hebbian,
                "discrete":LearningRule.discrete,
                "gradient":LearningRule.gradient
            }
        return super().setUp()
    
    def test_forward(self):
        '''
        Computes the outputs/activation of each neuron in the layer
        '''
        for case in self.test_cases["test_forward"].values():
            layer   = Layer(
                num_neurons = case["num_neurons"],
                num_inputs  = case["num_inputs"],
                active      = self.active[case["active"]],
                learn       = self.learn[case["learn"]]
            )

            for n, neuron in enumerate(case["neurons"].values()):
                # Initial weights for each neuron
                layer.neurons[n].weights    = neuron["weights"]
                
                # Assertion sequence of outputs
                outputs = []

                # TODO: forward pass for each input vector.
                for vec in neuron["inputs"]:
                    outputs.append(layer.forward(inputs=vec))

                for o, output in enumerate(outputs):
                    self.assertSequenceAlmostEqual(first=output, second=neuron["expected"][o])

        return
    
if __name__ == "__main__":
    unittest.main()