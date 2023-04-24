import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from io import BytesIO


class Environment:
    def __init__(self, num_of_state_types: int, state_types: List[int],
                 num_observe_types: int, observe_probs: np.ndarray,
                 action_effects: List[Dict], transition_matrices: np.ndarray, init_probs: np.ndarray):
        '''
        Initialize a representation of the environment. 
        Set attributes of the environment and initialize the state at time step 0.

        :param num_of_state_types: Number of state types.
        :param state_types: List of state types.

        :param num_observe_types: Number of observation types.
        :param observe_probs: List of Dicts describing the probabilities of 
            generating the observation types for each state type.

        :param action_effects: List of Dicts describing the effects of the actions.
        :param transition_matrices: A matrix representing the probabilities of 
            transitioning from one state to another.

        :param init_probs: the initial probability distribution    
        '''

        # Check input requirements
        self.num_state_types = num_of_state_types

        for state_type in state_types:
            assert 0 <= state_type <= self.num_state_types - \
                1,  "Each state type must be in [0, num_state_types - 1]"
        assert len(
            observe_probs) == self.num_state_types,       "Length of observe_probs should be equal to the number of state types"
        for probs in observe_probs:
            assert len(
                probs) == num_observe_types,         "Length of each element in observe_probs should be equal to the number of observation types"

        self.state_types = state_types
        self.num_observe_types = num_observe_types

        # infer # of states from the state_types list
        self.num_states = len(state_types)

        self.init_probs = init_probs

        # We will use this to generate the observation matrix
        self.observe_probs = observe_probs
        self.observation_matrices = None
        self.create_observation_matrices()

        # We will use these to generate the transition matrices
        self.action_effects = action_effects
        self.transition_matrices = transition_matrices
        if self.action_effects is not None:
            self.num_actions = len(action_effects)
        else:
            self.num_actions = len(transition_matrices)

        # ======================================================================
        # Initialize the state for time 0 randomly
        self.__pos = np.random.randint(0, self.num_states - 1)

        # Keep track of the sequence of states in the past
        self.__trajectory = [self.__pos]


    def create_observation_matrices(self):
        self.observation_matrices = []
        for i in range(self.num_states):
            self.observation_matrices.append(
                self.observe_probs[self.state_types[i]])
        self.observation_matrices = np.array(self.observation_matrices)


    def transition_dist(self, action) -> np.ndarray:
        """transition_dist: return the transition distribution P(S_k|S_k-1) given action at k-1

        Args:
            action (int or None): action at k-1 

        Returns:
            np.ndarray: P(S_k | S_k-1) with shape (num_states, num_states)
        """
        if action is None:
            action = 0
        return self.transition_matrices[action]


    def observation_dist(self, observation: int) -> np.ndarray:
        """observation_dist return the transition distribution P(o_k|S_k) given observation at o_k

        Args:
            observation (int): observation at k 

        Returns:
            np.array: P(o_k | S_k) with shape (num_states, )
        """

        return self.observation_matrices.T[observation]

    # ==========================================================================
    # helpful for simulating an example robot environment
    # ==========================================================================

    def observe(self) -> int:
        '''
        Returns an observation at the current time step.
        :return: An observation at the current time step.
        '''
        state_type = self.state_types[self.__pos]
        distribution = self.observe_probs[state_type]

        observation = np.random.choice(
            range(len(distribution)), p=distribution)
        return observation


    def move(self, action: int):
        '''
        Simulate the action according to the transition probabilities for the action
        '''
        distribution = self.action_effects[action]
        state_offset = np.random.choice(
            list(distribution.keys()), p=list(distribution.values()))

        self.__pos = (self.__pos + state_offset) % self.num_states
        self.__trajectory.append(self.__pos)


    def get_cur_pos(self) -> int:
        '''
        Returns the current state
        :return: The current state
        '''
        return self.__pos


    def get_past_pos(self, k: int) -> int:
        '''
        Returns the state at time step k in the past
        :return: The state at time step k in the past
        '''
        assert k < len(
            self.__trajectory), "k must be less than the number of past positions."
        return self.__trajectory[k]


    def act_and_observe(self, actions: List[int]) -> List[int]:
        '''
        Simulate the effect of taking the provided list of actions in the environment, 
        returning a list of observations.
        An initial observation is collected before actions are applied. 
        If you pass a list of n actions, a list of n + 1 observations are returned.

        :param actions: A list of actions. 
                        For example: [1, 1, 2, 0]
        :return: A list of observations.
                 For example: [0, 1, 1, 1, 1]
        '''

        assert all(a in range(self.num_actions)
                   for a in actions), "One or more actions are invalid."

        # Alternate between collecting an observation and taking an action
        observations = []
        # Collect observation in initial state
        observations.append(self.observe())

        # Apply each action, collecting an observation after each transition
        for a in actions:
            self.move(a)  # Apply the action
            observations.append(self.observe())   # Collect an observation
        return observations


def visualize_belief(env: Environment, probs: Dict, k=None):
    '''
    Plot the current state and the agent's probabilistic beliefs 
    over the states at time k.

    Yellow bars are the states of type 0. 
    The red rectangle is the state that the agent is in at time step k.

    :param env: The environment.
    :param probs: The agent's beliefs over the states at time k.
    :param k: The time step at which to plot the state. 
    '''
    fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]})

    # Plot the agent's beliefs for the state at time k.
    locs = list(range(env.num_states))
    ax[0].bar(locs, probs)

    ax[0].set_xlabel('Locations')
    ax[0].set_ylabel('Location Belief Probabilities')

    ax[0].set_xticks(np.arange(0, env.num_states, 1))
    ax[0].set_ylim([0., 1.])

    # Plot the states of type 0
    states_type_zero = np.zeros((env.num_states))
    for i in range(env.num_states):
        if env.state_types[i] == 0:
            states_type_zero[i] = 1.
    ax[1].bar(locs, states_type_zero, color='yyyy', label='states of type 0')

    # Plot the state that the agent is in at time step k.
    curr_state = env.get_cur_pos() if k is None else env.get_past_pos(k)
    ax[1].bar(curr_state, 0.5, color='r', label='current state')
    ax[1].legend(prop={"size": 8})

    ax[1].set_xlabel('Locations')
    ax[1].set_ylim([0., 2.])

    ax[1].set_xticks(np.arange(0, env.num_states, 1))
    ax[1].set_yticks([])

    plt.tight_layout()
    plt.show()

