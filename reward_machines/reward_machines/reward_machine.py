#Adapted from 
#https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/reward_machines/reward_machine.py

from reward_machines.reward_machines.reward_functions import *
from reward_machines.reward_machines.reward_machine_utils import evaluate_dnf 

class RewardMachine:
    def __init__(self, file):
        # <U,u0,delta_u,delta_r>
        self.U  = []         # list of non-terminal RM states
        self.u0 = None       # initial state
        self.delta_u    = {} # state-transition function
        self.delta_r    = {} # reward-transition function
        self.terminal_u = {}
        #self.terminal_u = -1  # If fall off RM sent to terminal state with id *-1*
        self._load_reward_machine(file)
        self.known_transitions = {} # Auxiliary variable to speed up computation of the next RM state

    # Public methods -----------------------------------

    def reset(self):
        # Returns the initial state
        return self.u0

    def _compute_next_state(self, u1, true_props):
        for u2 in self.delta_u[u1]:
            if evaluate_dnf(self.delta_u[u1][u2], true_props):
                return u2
        #Always expect transition to be defined for true_props #Doesn't care if makes other props true throughout 
        #Need to define NOT(2nd proposition) in a transition for 1st prop (to self and 2nd prop) if we care about the order that the task is done? Maybe this is why want to be able to send all transitions to terminal state. 
        #return self.terminal_u # no transition is defined for true_props

    def get_next_state(self, u1, true_props):
        if (u1,true_props) not in self.known_transitions:
            u2 = self._compute_next_state(u1, true_props)
            self.known_transitions[(u1,true_props)] = u2
        return self.known_transitions[(u1,true_props)]

    def step(self, u1, true_props, s_info, env_done=False):
        """
        Emulates an step on the reward machine from state *u1* when observing *true_props*.
        The rest of the parameters are for computing the reward when working with non-simple RMs: s_info (extra state information to compute the reward).
        """

        # Computing the next state in the RM and checking if the episode is done
        assert u1 not in self.terminal_u, "the RM was set to a terminal state!"
        u2 = self.get_next_state(u1, true_props)
        done = (u2 in self.terminal_u)
        # Getting the reward
        rew = self._get_reward(u1,u2,s_info, env_done)

        return u2, rew, done


    def get_states(self):
        return self.U

    def get_useful_transitions(self, u1):
        # This is an auxiliary method used by the HRL baseline to prune "useless" options
        return [self.delta_u[u1][u2].split("&") for u2 in self.delta_u[u1] if u1 != u2]


    # Private methods -----------------------------------

    def _get_reward(self,u1,u2,s_info,env_done):
        """
        Returns the reward associated to this transition.
        """
        # Getting reward from the RM
        reward = 0 # NOTE: if the agent falls from the reward machine it receives reward of zero 
        #NOTE: what does "falls from RM" mean? Looks like it's if agent makes a transition that shouldn't be possible 
        if u1 in self.delta_r and u2 in self.delta_r[u1]:
            reward += self.delta_r[u1][u2].get_reward(s_info)
         # Returning final reward
        return reward 


    def _load_reward_machine(self, file):
        """
        Example:
            0      # initial state
            [2]    # terminal state
            (0,0,'!e&!n',ConstantRewardFunction(0))
            (0,1,'e&!g&!n',ConstantRewardFunction(0))
            (0,2,'e&g&!n',ConstantRewardFunction(1))
            (1,1,'!g&!n',ConstantRewardFunction(0))
            (1,2,'g&!n',ConstantRewardFunction(1))
        """
        # Reading the file
        f = open(file)
        lines = [l.rstrip() for l in f]
        f.close()
        # setting the DFA
        self.u0 = eval(lines[0])
        self.terminal_u = eval(lines[1])
        # adding transitions
        for e in lines[2:]:
            # Reading the transition
            u1, u2, dnf_formula, reward_function = eval(e)
            # terminal states
        #    if u1 in terminal_states:
        #        continue
         #   if u2 in terminal_states:
         #       u2  = self.terminal_u
            # Adding machine state
            self._add_state([u1,u2])
            # Adding state-transition to delta_u
            if u1 not in self.delta_u:
                self.delta_u[u1] = {}
            self.delta_u[u1][u2] = dnf_formula
            # Adding reward-transition to delta_r
            if u1 not in self.delta_r:
                self.delta_r[u1] = {}
            self.delta_r[u1][u2] = reward_function
        # Sorting self.U... just because... 
        self.U = sorted(self.U)

    def _add_state(self, u_list):
        for u in u_list:
            if u not in self.U and u not in self.terminal_u:
                self.U.append(u)