import json
import unittest
import numpy as np
from cartpole.cartPole import CartPole

class TestCartPole(unittest.TestCase):

    def setUp(self):
        self.cart_pole = CartPole()

    def test_load_default_parameters(self):
        parameters = self.cart_pole.load_parameters(parameters_path="non_existent_file.json")
        
        # Check if the parameters are correctly loaded
        self.assertIsInstance(parameters, dict)
        self.assertEqual(len(parameters), 12)
        self.assertEqual(parameters['CART_VELOCITY_MIN'], -3)
        self.assertEqual(parameters['CART_VELOCITY_MAX'], 3)
        self.assertEqual(parameters['POLE_ANGLE_VELOCITY_MIN'], -10)
        self.assertEqual(parameters['POLE_ANGLE_VELOCITY_MAX'], 10)
        self.assertEqual(parameters['NUMBER_OF_BIN_POSITION'], 30)
        self.assertEqual(parameters['NUMBER_OF_BIN_VELOCITY'], 30)
        self.assertEqual(parameters['NUMBER_OF_BIN_ANGLE'], 30)
        self.assertEqual(parameters['NUMBER_OF_BIN_ANGLE_VELOCITY'], 30)
        self.assertEqual(parameters['ALPHA'], 0.1)
        self.assertEqual(parameters['GAMMA'], 1)
        self.assertEqual(parameters['EPSILON'], 0.2)
        self.assertEqual(parameters['NUMBER_OF_EPOCH'], 15000)
    
    def test_load_parameters_from_file(self):
        parameters = self.cart_pole.load_parameters(parameters_path=CartPole.cart_pole_parameters_path)
        
        param = None
        with open(CartPole.cart_pole_parameters_path, 'r') as file:
            param = json.load(file)
            
        # Check if the parameters are correctly loaded
        self.assertIsInstance(parameters, dict)
        self.assertEqual(len(parameters), 12)
        self.assertEqual(parameters, param)

    def test_get_bounds(self):
        bounds = self.cart_pole.get_bounds()
        
        # Check if the bounds are correctly created
        self.assertIsInstance(bounds, dict)
        self.assertEqual(len(bounds), 2)
        self.assertIsInstance(bounds['upperBounds'], np.ndarray)
        self.assertIsInstance(bounds['lowerBounds'], np.ndarray)
        
        # Check if the bounds are correctly set
        self.assertEqual(bounds['upperBounds'][1], self.cart_pole.parameters['CART_VELOCITY_MAX'])
        self.assertEqual(bounds['upperBounds'][3], self.cart_pole.parameters['CART_VELOCITY_MIN'])
        self.assertEqual(bounds['lowerBounds'][1], self.cart_pole.parameters['POLE_ANGLE_VELOCITY_MIN'])
        self.assertEqual(bounds['lowerBounds'][3], self.cart_pole.parameters['POLE_ANGLE_VELOCITY_MAX'])
  
    def test_discrete_states(self):
        discrete_states = self.cart_pole.discrete_states()
        
        # Check if the discrete states are numpy arrays
        self.assertIsInstance(discrete_states, tuple)
        self.assertEqual(len(discrete_states), 4)
        for state in discrete_states:
            self.assertIsInstance(state, np.ndarray)
        
        # Check if the bins are correctly created
        self.assertEqual(len(discrete_states[0]), self.cart_pole.parameters['NUMBER_OF_BIN_POSITION'])
        self.assertEqual(len(discrete_states[1]), self.cart_pole.parameters['NUMBER_OF_BIN_VELOCITY'])
        self.assertEqual(len(discrete_states[2]), self.cart_pole.parameters['NUMBER_OF_BIN_ANGLE'])
        self.assertEqual(len(discrete_states[3]), self.cart_pole.parameters['NUMBER_OF_BIN_ANGLE_VELOCITY'])
        
        # Check if the bins cover the correct range
        self.assertAlmostEqual(discrete_states[0][0], self.cart_pole.bounds['lowerBounds'][0])
        self.assertAlmostEqual(discrete_states[0][-1], self.cart_pole.bounds['upperBounds'][0])
        self.assertAlmostEqual(discrete_states[1][0], self.cart_pole.bounds['lowerBounds'][1])
        self.assertAlmostEqual(discrete_states[1][-1], self.cart_pole.bounds['upperBounds'][1])
        self.assertAlmostEqual(discrete_states[2][0], self.cart_pole.bounds['lowerBounds'][2])
        self.assertAlmostEqual(discrete_states[2][-1], self.cart_pole.bounds['upperBounds'][2])
        self.assertAlmostEqual(discrete_states[3][0], self.cart_pole.bounds['lowerBounds'][3])
        self.assertAlmostEqual(discrete_states[3][-1], self.cart_pole.bounds['upperBounds'][3])

if __name__ == '__main__':
    unittest.main()