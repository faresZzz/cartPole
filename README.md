# CS433 Machine Learning Project 1 - Group Project

## Table of Contents

1. [Introduction](#introduction)
2. [Running the Scripts](#running-the-scripts)
3. [Project Structure](#project-structure)
4. [Contributors](#contributors)

---

## Introduction

The Cart-Pole Q-Learning project demonstrates the application of reinforcement learning to solve the classic Cart-Pole control problem. Using the Q-learning algorithm, the agent learns to balance a pole on a moving cart by applying horizontal forces. This project provides an implementation of Q-learning, highlights its key concepts, and showcases how the agent evolves through training to maximize its performance in the environment.
![Cart-Pole Environment](fig/cartpole.png)


## Running the Scripts

Please ensure that you have the necessary dependencies installed before running the scripts by executing the following
command:

```shell
pip install -r requirements.txt
```


0. **main.py** - This script is for running the whole pipeline.
1. **tests** - This script is for running the test d

## Project Structure

```

CartPole/
│
├── cart_pole/  project material
│   ├── __init__.py/  
│   ├── CartPole.py/ # Cart Pole env handeling  
│   ├── main.py/ # entry point 
│   ├── Q_learning.py/  
│   └── utils.py/ #utilis fonction
│
├── config/  # all the config files of the project
│   └── parameters.json/ # config file containing the parameters
│
├── fig/  # directory to plot the figures
│
├── models/  # directory to store the trained models
│   
├── tests/ # test folder
│   ├── __init__.py/  
│   ├── test_CartPole.py/ 
│   └── test_Q_learning.py/ 
│   
├── .gitignore
├── README.md # This file
└── requirements.txt # Python dependencies
```


## Contributors

- [Zaghouane Fares](https://github.com/faresZzz)