import numpy as np
import sys
from gym.envs.toy_text import discrete

"""
 
 Creates an environment for Example 13.1 from Reinforcement Learning: An Introduction
 by Sutton and Barto:
 http://incompleteideas.net/book/RLbook2020.pdf

 Adapted from cliffwalking in openai-gym

 """


class ShortCorridorEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):  # __init__(self, n_starting_states =100,max_delta=10):
        self.start_state_index = 0
        self.shape = (1, 4)
        nS = np.prod(self.shape)
        nA = 2

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}
            for a in range(nA):
                P[s][a] = self._calculate_transition_prob(s, a)

        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(ShortCorridorEnv, self).__init__(nS, nA, P, isd)


    def _calculate_transition_prob(self, state, action):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """
        if state != 1:
            new_state = state-np.power(-1,action) # action 0 left, action 1 right
            new_state=max(0,new_state)
            new_state = min(self.shape[1]-1, new_state)
        else: # reverse for state 1
            new_state = state + np.power(-1, action)

        terminal_state_R = self.shape[1]-1
        is_done = new_state == terminal_state_R

        if is_done:
            return [(1.0, new_state, -1, is_done)]

        return [(1.0, new_state, -1, False)]

    def render(self, mode='human'):
        outfile = sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif position == (0, self.shape[1] - 1):
                output = " G "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')