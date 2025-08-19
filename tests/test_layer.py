import unittest
import json

from pathlib import Path
from test_setup import *
from neuron import *
from layer import *

class TestLayer(unittest.TestCase):
    # helper for comparing float sequences
    def assertSequenceAlmostEqual(self, first, second, places=6):
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
        for case in self.test_cases["test_forward"]:
            layer   = Layer(
                num_neurons = case["num_neurons"],
                num_inputs  = case["num_inputs"],
                active      = self.active[case["active"]],
                learn       = self.learn[case["learn"]]
            )

            for n, neuron in enumerate(case["neurons"]):
                # Initial weights for each neuron
                layer.neurons[n].weights    = neuron["weights"]
                
                # TODO: forward pass for each input vector.
        return