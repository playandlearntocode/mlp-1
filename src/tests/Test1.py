import unittest
from classes.MLP import MLP

class Test1(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_activation_function(self):
        x = 0.5
        mlp = MLP([])
        ret = mlp.activation_function(1000)
        self.assertTrue(ret <= 1.00 and ret >= -1.00)

        ret = mlp.activation_function(-10)
        self.assertTrue(ret <= 1.00 and ret >= -1.00)
