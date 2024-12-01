import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt 
import json

from cartpole.utils import Strategy, ActionStrategy
from cartpole.Q_learning import Q_Learning

class CartPole: 
    cart_pole_parameters_path = './config/parameters.json'
    def __init__(self):
        # Create CartPole environment with no rendering
        self.env = gym.make('CartPole-v1', render_mode=None)
        self.parameters = self.load_parameters()
        self.bounds = self.get_bounds()
        self.cart_pole_discrete_states = self.discrete_states()

        self.Q_learning = Q_Learning(
            actionNumber=self.env.action_space.n ,
            ALPHA=self.parameters['ALPHA'], 
            GAMMA=self.parameters['GAMMA'], 
            EPSILON=self.parameters['EPSILON'], 
            numberOfBins=[self.parameters['NUMBER_OF_BIN_POSITION'], 
                        self.parameters['NUMBER_OF_BIN_VELOCITY'], 
                        self.parameters['NUMBER_OF_BIN_ANGLE'], 
                        self.parameters['NUMBER_OF_BIN_ANGLE_VELOCITY']
                        ])
        
    def load_parameters(self, parameters_path : str= cart_pole_parameters_path):
        # load parameters form file if exist or set of default parametes
        try:
            with open(parameters_path, 'r') as file:
                parameters = json.load(file)
        except FileNotFoundError:
            parameters = self.default_parameters()
            
        return parameters

    def default_parameters(self):

        return {
            # ENVIRONMENT PARAMETERS
            "CART_VELOCITY_MIN": -3,
            "CART_VELOCITY_MAX": 3,
            "POLE_ANGLE_VELOCITY_MIN": -10,
            "POLE_ANGLE_VELOCITY_MAX": 10,

            # AGENT PARAMETERS
            "NUMBER_OF_BIN_POSITION": 30,
            "NUMBER_OF_BIN_VELOCITY": 30,
            "NUMBER_OF_BIN_ANGLE": 30,
            "NUMBER_OF_BIN_ANGLE_VELOCITY": 30,

            # Q-LEARNING PARAMETERS
            "ALPHA": 0.1,
            "GAMMA": 1,
            "EPSILON": 0.2,
            "NUMBER_OF_EPOCH": 15000
        }
    
    def get_bounds(self):
        # set bound using env boundaries and parameters ( due to some infinit boudaries in the env)
        bound  = {}
        upperBounds = self.env.observation_space.high
        lowerBounds = self.env.observation_space.low
        upperBounds[1] = self.parameters['CART_VELOCITY_MAX']
        upperBounds[3] = self.parameters['CART_VELOCITY_MIN']
        lowerBounds[1] = self.parameters['POLE_ANGLE_VELOCITY_MIN']
        lowerBounds[3] = self.parameters['POLE_ANGLE_VELOCITY_MAX']
        bound['upperBounds'] = upperBounds 
        bound['lowerBounds'] = lowerBounds
        return bound
    
    def discrete_states(self):
        # env state is a continuous state. 
        # we only have finite space so we need to dicretise the states
        cart_position_bins = np.linspace(self.bounds['lowerBounds'][0], self.bounds['upperBounds'][0], self.parameters['NUMBER_OF_BIN_POSITION'])
        cart_velocity_bins = np.linspace(self.bounds['lowerBounds'][1], self.bounds['upperBounds'][1], self.parameters['NUMBER_OF_BIN_VELOCITY'])
        pole_angle_bins = np.linspace(self.bounds['lowerBounds'][2], self.bounds['upperBounds'][2], self.parameters['NUMBER_OF_BIN_ANGLE'])
        pole_angle_velocity_bins = np.linspace(self.bounds['lowerBounds'][3], self.bounds['upperBounds'][3], self.parameters['NUMBER_OF_BIN_ANGLE_VELOCITY'])

        return (cart_position_bins, cart_velocity_bins, pole_angle_bins, pole_angle_velocity_bins)
    
    def discretize_state(self, state):
        # return the discretize value of the continuoue states
        # select the corresponding discretie value for each state variable
        cart_position = np.maximum(np.digitize(state[0], self.cart_pole_discrete_states[0]) - 1, 0)
        cart_velocity = np.maximum(np.digitize(state[1], self.cart_pole_discrete_states[1]) - 1, 0)
        pole_angle = np.maximum(np.digitize(state[2], self.cart_pole_discrete_states[2]) - 1, 0)
        pole_angle_velocity = np.maximum(np.digitize(state[3], self.cart_pole_discrete_states[3]) - 1, 0)
        
        return (cart_position, cart_velocity, pole_angle, pole_angle_velocity)
    
    def saveQTable(self, file_path = 'Q_table.npy'):
        # Save Q-table
        self.Q_learning.saveQTable(file_path)

    def select_action(self, states : list, action_strategy : ActionStrategy):
        # we can choose between 3 stratedy for selecting the next action
        # depending on the need
        if action_strategy == ActionStrategy.RANDOM:
            return self.Q_learning.select_action_random()
        elif action_strategy == ActionStrategy.GREEDY:
            return self.Q_learning.select_action_greedy(states)
        elif action_strategy == ActionStrategy.EPSILON_GREEDY:
            return self.Q_learning.select_action_epsilon_greedy(states)
        
    def train(self):
        print("Training model.")
        # Initialize action strategy to random to emphasize exploration
        action_strategy = ActionStrategy.RANDOM

        for epoch in range(1, self.parameters['NUMBER_OF_EPOCH']):
            # reset the env
            (state, _) = self.env.reset()
            done = False
            total_reward = 0
            rewards = []
            # after 500 episodes we switch to epsilon greedy policy 
            if epoch > 500:
                action_strategy = ActionStrategy.EPSILON_GREEDY
            # after 10000 episodes we decrease the epsilon to 0.1% to emphasize exploitation
            if epoch > 10_000:
                self.Q_learning.EPSILON = 0.999*self.Q_learning.EPSILON
        
            while not done:
                # discretize state
                discrete_state = self.discretize_state(state)

                # select the action base on the state and the policy 
                action = self.select_action(states=discrete_state, action_strategy=action_strategy)

                # Take action and observe next state and reward
                (next_state, reward, done, _, _) = self.env.step(action)
                next_discrete_state = self.discretize_state(next_state)
                
                # Update Q-table
                self.Q_learning.learn(reward, discrete_state, next_discrete_state, action, done)

                # Update state
                state = next_state
                total_reward += reward

            # Save rewards
            self.Q_learning.rewards.append(total_reward)
            rewards.append(total_reward)
            # Print rewards
            if epoch % 500 == 0:
                print(f"epoch: {epoch}, max_reward: {max(rewards)}, min_reward: {min(rewards)}, avg_reward: {sum(rewards)/len(rewards)}")
                rewards = []

    def simulate(self,  strategy: Strategy, file_path='Q_table.npy'):
        # Simulate the cartPole with rendering. 
        # Can be for showing random or learned strategy

        # Select the strategy for the Q_matrix if random, already trained, loading, need to train
        if strategy == Strategy.RANDOM:
            print("Applying random strategy.")
        elif strategy == Strategy.TRAINED:
            print("Applying already trained model.")
        elif strategy == Strategy.LOAD_AND_APPLY:
            print("Loading and applying trained model from file.")
            self.Q_learning.loadQTable(file_path)
        elif strategy == Strategy.TRAIN_AND_APPLY:
            print("Training model first, then applying it.")
            self.train()
            self.saveQTable()

        # create env with rendering 
        self.env = gym.make('CartPole-v1', render_mode='human')
        (state, _) = self.env.reset()
        done = False
        total_reward = 0

        # Run an epoch
        while not done:
            self.env.render()
            discrete_state = self.discretize_state(state)

            # simulate learned strategy or random strategy
            if strategy == Strategy.RANDOM: 
                action = self.select_action(states=discrete_state, action_strategy=ActionStrategy.RANDOM)
            else:
                # Select action using greedy policy
                action = self.select_action(states = discrete_state, action_strategy=ActionStrategy.GREEDY)

            (state, reward, done, _, _) = self.env.step(action)

            total_reward += reward

        print(f"Total reward: {total_reward}")
        self.env.close()
    
    

