from enum import Enum
import matplotlib.pyplot as plt


class Strategy(Enum):
    RANDOM = "random" # Use random actions ( use it to compare performances of our trained algo over randomness)
    TRAINED = "trained" #Use the already trained Q-table 
    LOAD_AND_APPLY = "load_and_apply" # Use one trained Q-table from saved file
    TRAIN_AND_APPLY = "train_and_apply" # Tain and simulate the Q-table 

class ActionStrategy(Enum):
    RANDOM = "random"
    GREEDY = "greedy"
    EPSILON_GREEDY = "epsilon_greedy"



def plot_rewards(rewards : list[float], epochs : int, file_path : str = "."):
        plt.figure(figsize=(12, 5))
        plt.plot(rewards, color='blue', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Sum of Rewards in Epoch')
        plt.yscale('log')
        plt.savefig(f"{file_path}convergence_{epochs}_epoch.png")
        plt.show()