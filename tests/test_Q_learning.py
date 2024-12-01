import unittest
import numpy as np
from cartpole.Q_learning import Q_Learning
import gymnasium as gym

class TestQLearning(unittest.TestCase):
    
    def setUp(self):
        self.env = gym.make('CartPole-v1')
        self.q_learning = Q_Learning(actionNumber=self.env.action_space.n)

    def test_initialization(self):
        self.assertEqual(self.q_learning.actionNumber, self.env.action_space.n)
        self.assertEqual(self.q_learning.ALPHA, 0.1)
        self.assertEqual(self.q_learning.GAMMA, 0.99)
        self.assertEqual(self.q_learning.EPSILON, 0.2)
        self.assertEqual(self.q_learning.Q_matrix.shape, (30, 30, 30, 30, self.q_learning.actionNumber))
    
    def test_save_and_load_Q_table(self):
        self.q_learning.saveQTable('test_Q_table.npy')
        original_Q_matrix = self.q_learning.Q_matrix.copy()
        self.q_learning.Q_matrix = np.zeros_like(self.q_learning.Q_matrix)
        self.q_learning.loadQTable('test_Q_table.npy')
        np.testing.assert_array_equal(self.q_learning.Q_matrix, original_Q_matrix)
    
    def test_select_action_epsilon_greedy(self):
        state = [0, 0, 0, 0]
        action = self.q_learning.select_action_epsilon_greedy(state)
        self.assertIn(action, range(self.q_learning.actionNumber))
    
    def test_select_action_random(self):
        action = self.q_learning.select_action_random()
        self.assertIn(action, range(self.env.action_space.n +1))
    
    def test_select_action_greedy(self):
        state = (0, 0, 0, 0)
        action = self.q_learning.select_action_greedy(state)
        self.assertIn(action, (i for i in range(self.q_learning.actionNumber)))
    
    def test_learn(self):
        state = (0, 0, 0, 0)
        next_state = (0, 0, 0, 1)
        action = 1
        reward = 1.0
        done = False
        old_value = self.q_learning.Q_matrix[state + (action,)]
        self.q_learning.learn(reward, state, next_state, action, done)
        new_value = self.q_learning.Q_matrix[state + (action,)]
        self.assertNotEqual(old_value, new_value)

if __name__ == '__main__':
    unittest.main()