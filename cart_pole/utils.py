from enum import Enum
import matplotlib as plt

from cartPole import CartPole

class Strategy(Enum):
    RANDOM = "random"
    TRAINED = "trained"
    LOAD_AND_APPLY = "load_and_apply"
    TRAIN_AND_APPLY = "train_and_apply"

class ActionStrategy(Enum):
    RANDOM = "random"
    GREEDY = "greedy"
    EPSILON_GREEDY = "epsilon_greedy"



def plot_rewards(cartPole: CartPole, file_path : str = "."):
        plt.figure(figsize=(12, 5))
        plt.plot(cartPole.Q_learning.rewards, color='blue', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Sum of Rewards in Epoch')
        plt.yscale('log')
        plt.savefig(f"{file_path}convergence_{cartPole.parameters['NUMBER_OF_EPOCH']}_epoch.png")
        plt.show()