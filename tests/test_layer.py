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
        return super().setUp()
    
    def test_forward(self):
        return