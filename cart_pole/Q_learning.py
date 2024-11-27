import numpy as np


class Q_Learning:
    def __init__(self,  actionNumber : int, ALPHA : float = 0.1, GAMMA : float = 0.99 , EPSILON : float = 0.2, numberOfBins : list = [30,30,30,30]):
        # init Q_learning model
        self.actionNumber = actionNumber
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.EPSILON = EPSILON
        self.rewards = []
        self.Q_matrix = np.random.uniform(low=0.0, high=1.0, size=(numberOfBins[0],numberOfBins[1],numberOfBins[2],numberOfBins[3],self.actionNumber))

    def saveQTable(self, file_path : str = 'Q_table.npy'):
        np.save(file_path, self.Q_matrix)
    
    def loadQTable(self, file_path : str = 'Q_table.npy'):
        try:
            self.Q_matrix = np.load(file_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {file_path} not found.") from e
    
    def select_action_epsilon_greedy(self, state : list):
        # select and action under epsilon-greedy policy
        if np.random.uniform(0,1) < self.EPSILON:
            return self.select_action_random()
        else:
            return self.select_action_greedy(state)
        
    def select_action_random(self):
        # select random action 
        return np.random.choice(self.actionNumber)
    
    def select_action_greedy(self, state : list):
        # select action under greedy policy
        return np.random.choice(np.where(self.Q_matrix[state]==np.max(self.Q_matrix[state]))[0])

    def learn(self, reward : float, state : list, next_state : list, action : int, done : bool):
        # update Q_matrix[Q(s,a)]

        if not done:
            # select Q_max = max(Q(s',a))
            Q_max_prime = np.max(self.Q_matrix[next_state])
        else: 
            # by definition when done 
            Q_max_prime = 0

        # state index
        state_index = state+(action,)
        #compute error 
        error=reward+self.GAMMA*Q_max_prime-self.Q_matrix[state_index]
        #update Q(s,a)
        self.Q_matrix[state_index]=self.Q_matrix[state_index]+self.ALPHA*error

    

