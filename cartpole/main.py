"""
Main module for training, saving, simulating, and plotting rewards for the CartPole environment.

This script performs the following steps:
1. Initializes the CartPole environment.
2. Trains the CartPole agent.
3. Saves the Q-table after training.
4. Simulates the CartPole environment using the trained strategy.
5. Plots the rewards obtained during training.

Functions:
    main(): Main function to execute the above steps.

Usage:
    Run this script directly to train, save, simulate, and plot rewards for the CartPole environment.

"""

from cartpole.cartPole import CartPole
from cartpole.utils import Strategy, plot_rewards


def main():
    cart_pole = CartPole()
    cart_pole.train()
    plot_rewards(file_path="./fig/")
    cart_pole.saveQTable(file_path = "./models/Q_table.npy")
    cart_pole.simulate(Strategy.TRAINED, file_path = "./models/Q_table.npy")

if __name__ == "__main__":
    main()

