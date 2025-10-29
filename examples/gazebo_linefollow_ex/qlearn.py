import random
import pickle
import os

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        # TODO: Implement loading Q values from pickle file.
        # Load stored values
        if os.path.isfile(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'rb') as f:  # binary read
                self.q = pickle.load(f)
            print(f"Loaded file: {filename}")
        else:
            print(f"No Q-values to load from {filename}. Starting fresh.")

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.
        with open(filename, 'wb') as f:
            pickle.dump(self.q, f)

        print(f"Wrote to file: {filename}")

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 
        q_values = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q_values)

        # Exploration vs Exploitation
        if random.random() < self.epsilon:
            action = random.choice(self.actions)  # proper random action
        else:
            # Exploitation
            max_indices = [i for i, q in enumerate(q_values) if q == maxQ]
            i = random.choice(max_indices)
            action = self.actions[i]

        if return_q:
            return action, q_values  # for debugging or plotting
        return action                # for normal use


    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

    def learnQ(self, state, action, reward, value):
        # Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        oldv = self.q.get((state, action), None)

        if oldv is None:
            self.q[(state, action)] = reward + self.gamma * max([self.getQ(state, a) for a in self.actions])
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)
