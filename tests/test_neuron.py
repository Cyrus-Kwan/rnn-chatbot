import unittest
import json

from pathlib import Path
from test_setup import *
from neuron import *

class TestNeuron(unittest.TestCase):
    # helper for comparing float sequences
    def assertSequenceAlmostEqual(self, first, second, places=6):
        self.assertEqual(len(first), len(second), "Lengths differ")
        for i, (a, b) in enumerate(zip(first, second)):
            with self.subTest(index=i):
                self.assertAlmostEqual(a, b, places=places)

    def setUp(self):
        cur_path:Path   = Path(__file__).parent.resolve()
        with open(cur_path / "test_neuron.json") as f:
            self.test_cases:dict    = json.load(f)
        return super().setUp()

    def test_scalar(self):
        inputs      = self.test_cases["test_scalar"]["inputs"]
        weights     = self.test_cases["test_scalar"]["weights"]
        expected    = self.test_cases["test_scalar"]["expected"]
        for i, w, e in zip(inputs, weights, expected):
            with self.subTest(inputs=i, weights=w):
                n   = Neuron(
                    inputs=i, 
                    weights=w, 
                    active=Activation.sign_zero, 
                    learn=None
                )
                result  = n.scalar()
                self.assertAlmostEqual(first=result, second=e, places=3)

    def test_sign_zero(self):
        inputs      = self.test_cases["test_sign_zero"]["inputs"]
        weights     = self.test_cases["test_sign_zero"]["weights"]
        expected    = self.test_cases["test_sign_zero"]["expected"]
        for i, w, e in zip(inputs, weights, expected):
            with self.subTest(inputs=i, weights=w):
                n   = Neuron(
                    inputs=i, 
                    weights=w, 
                    active=Activation.sign_zero, 
                    learn=None
                )
                result  = n.active(n)
                self.assertAlmostEqual(first=result, second=e, places=3)

    def test_sign_one(self):
        inputs      = self.test_cases["test_sign_one"]["inputs"]
        weights     = self.test_cases["test_sign_one"]["weights"]
        expected    = self.test_cases["test_sign_one"]["expected"]
        for i, w, e in zip(inputs, weights, expected):
            with self.subTest(inputs=i, weights=w):
                n   = Neuron(
                    inputs=i, 
                    weights=w, 
                    active=Activation.sign_one, 
                    learn=None
                )
                result  = n.active(n)
                self.assertAlmostEqual(first=result, second=e, places=3)

    def test_logistic(self):
        inputs      = self.test_cases["test_logistic"]["inputs"]
        weights     = self.test_cases["test_logistic"]["weights"]
        expected    = self.test_cases["test_logistic"]["expected"]
        for i, w, e in zip(inputs, weights, expected):
            with self.subTest(inputs=i, weights=w):
                n   = Neuron(
                    inputs=i, 
                    weights=w, 
                    active=Activation.logistic, 
                    learn=None
                )
                result  = n.active(n)
                self.assertAlmostEqual(first=result, second=e, places=3)

    def test_bipolar(self):
            inputs      = self.test_cases["test_bipolar"]["inputs"]
            weights     = self.test_cases["test_bipolar"]["weights"]
            expected    = self.test_cases["test_bipolar"]["expected"]
            for i, w, e in zip(inputs, weights, expected):
                with self.subTest(inputs=i, weights=w):
                    n   = Neuron(
                        inputs=i, 
                        weights=w, 
                        active=Activation.bipolar, 
                        learn=None
                    )
                    result  = n.active(n)
                    self.assertAlmostEqual(first=result, second=e, places=3)

    def test_hyper_log(self):
            inputs      = self.test_cases["test_hyper_log"]["inputs"]
            weights     = self.test_cases["test_hyper_log"]["weights"]
            expected    = self.test_cases["test_hyper_log"]["expected"]
            for i, w, e in zip(inputs, weights, expected):
                with self.subTest(inputs=i, weights=w):
                    n   = Neuron(
                        inputs=i, 
                        weights=w, 
                        active=Activation.hyper_log, 
                        learn=None
                    )
                    result  = n.active(n)
                    self.assertAlmostEqual(first=result, second=e, places=3)

    def test_discrete(self):
        inputs      = self.test_cases["test_discrete"]["inputs"]
        weights     = self.test_cases["test_discrete"]["weights"]
        desired     = self.test_cases["test_discrete"]["desired"]
        expected    = self.test_cases["test_discrete"]["expected"]

        n   = Neuron(
            inputs=None,
            weights=weights,
            active=Activation.sign_one,
            learn=LearningRule.discrete
        )

        for cycle in range(10):
            for i in range(len(desired)):
                n.inputs    = inputs[i]
                n.learn(n, c=0.05, d=desired[i])

        self.assertSequenceAlmostEqual(first=n.weights, second=expected, places=3)

    def test_gradient(self):
        inputs      = self.test_cases["test_gradient"]["inputs"]
        weights     = self.test_cases["test_gradient"]["weights"]
        desired     = self.test_cases["test_gradient"]["desired"]
        expected    = self.test_cases["test_gradient"]["expected"]

        n   = Neuron(
            inputs=None,
            weights=weights,
            active=Activation.bipolar,
            learn=LearningRule.gradient
        )

        for cycle in range(20):
            for i in range(len(desired)):
                n.inputs    = inputs[i]
                n.learn(n, c=0.2, d=desired[i])

        self.assertSequenceAlmostEqual(first=n.weights, second=expected, places=3)

if __name__ == "__main__":
    unittest.main()