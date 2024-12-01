import unittest
from test_cartPole import TestCartPole
from test_Q_learning import TestQLearning

class Tests(unittest.TestCase):
    def setUp(self):
        self.tester_cartPole = TestCartPole()
        self.tester_Qlearning = TestQLearning()


if __name__ == '__main__':
    unittest.main()