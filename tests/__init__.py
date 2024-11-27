import sys
from os.path import dirname, abspath
# from .test_cartPole import TestCartPole
# from .test_Q_learning import TestQLearning


# add module dir to PYTHONPATH
_dir = dirname(dirname(abspath(__file__)))

sys.path.append(_dir + "/cart_pole")
