# -*- coding: utf-8 -*-
#
# Copyright 2023 University of Waterloo
# Author Y. Y. 
#
# This file is part of course material for CS458/658.


import numpy as np
from env import *
from typing import List, Tuple



def create_transition_matrices(env:Environment):
    '''
    If the transition_matrices in env is None,
    constructs a 3D numpy array containing the transition matrix for each action.
    Entry (i,j,k) is the probability of transitioning from state j to k
    given that the agent takes action i.
    Saves the matrices in env.transition_matrices and returns nothing.
    '''
    n = env.num_states
    res = []
    if env.transition_matrices is None:
        for action in env.action_effects:
            curr_matrix = []
            for state in range(n):
                curr = [0] * n
                for offset, prob in action.items():
                    ind = (state + offset) % n
                    curr[ind] = prob
                curr_matrix.append(curr.copy())
            res.append(curr_matrix.copy())
        env.transition_matrices = np.array(res)

        return



def forward(env: Environment, f: np.ndarray, action, observation: int) -> np.ndarray:
    """forward: perform HMM filtering through forward recursion at time step k 

    Args:
        env (Environment): the HMM environment
        f (np.array): message f0:k-1 at time step k - 1
        action (int or None): action performed at time step k - 1  
        observation (int): observation at time step k

    Returns:
        np.array: return normalized message f0:k at time step k, a 1D numpy array of shape (env.num_states,)
    """
    transition_matrix = env.transition_dist(action)
    raw_res = np.multiply(env.observation_dist(observation), np.dot(transition_matrix.T, f))
    summation = sum(raw_res)
    normalized_res = raw_res / summation
    return normalized_res



def backward(env: Environment, b: np.ndarray, action, observation: int) -> np.ndarray:
    """backward: perform HMM backward recursion at time step k

    Args:
        env (Environment): the HMM environment
        b (np.array): message bk+2:t-1 at time step k + 1
        action (_type_): action at time step k
        observation (int): observation at time step k + 1

    Returns:
        np.array: return message bk+1:t-1 at time step k, a 1D numpy array of shape (env.num_states,)
    """
    observe_matrix = env.observation_dist(observation)
    transition_matrix = env.transition_dist(action)

    obs_and_b = np.multiply(observe_matrix, b)

    # expand dimensions to take transpose to take dot product
    obs_and_b = np.expand_dims(obs_and_b, axis=0).T

    # take dot product of first part with transition matrix
    # reduce the dimension after obtaining result
    res = np.squeeze(np.dot(transition_matrix, obs_and_b).T)
    return res


def forward_backward(env: Environment, actions, observation: List[int]) -> np.ndarray:
    """forward_backward: perform HMM smoothing through forward-backward algorithm for each time step

    Args:
        env (Environment): the HMM enviroment 
        actions (List[int] or None): a list of actions from time step 0 to t - 2
        observation (List[int]): a list of observations from time step 0 to t - 1 

    Returns:
        np.array: A numpy array with shape (t, env.num_states)
        the k'th row represents the normalized smoothed probability distribution for time k.

    """
    t = len(observation)
    if actions is None:
        actions = [None] * t

    # forward base case
    raw_f_base = np.multiply(env.observation_dist(observation[0]), env.init_probs)
    s = sum(raw_f_base)
    f_base = raw_f_base / s
    fv = [f_base]
    curr_f = f_base

    # forward recursive case
    for k in range(1, t):
        next_f = forward(env, curr_f, actions[k - 1], observation[k])
        fv.append(next_f)
        curr_f = next_f

    # backward base case
    b_base = [1] * env.num_states
    bv = [b_base]
    curr_b = b_base

    # backward recursive case
    for k in reversed(range(0, t - 1)):
        next_b = backward(env, curr_b, actions[k], observation[k + 1])
        bv.append(next_b)
        curr_b = next_b

    # compute result
    res = []
    for i in range(t):
        f = fv[i]
        b = bv[t - i - 1]
        curr_raw_res = np.multiply(f, b)
        curr_normalized_res = curr_raw_res / sum(curr_raw_res)
        res.append(curr_normalized_res)

    ans = np.array(res)

    return ans

# Recursive case of viterbi
def viterbi_recurse(env: Environment, m, action, observation):
    transition_matrix = env.transition_dist(action)

    m = np.vstack([m for _ in range(env.num_states)]).T
    transition_matrix_m = np.multiply(transition_matrix, m)
    max_prob = np.max(transition_matrix_m, axis=0)

    states = np.argmax(transition_matrix_m, axis=0)

    observation_matrix = env.observation_dist(observation)

    res = np.multiply(observation_matrix, max_prob)

    return res, states

def viterbi(env: Environment, actions: List[int], observations: List[int]) -> Tuple[List, List]:
    """viterbi: find the most likely sequence with viterbi algorithm

    Args:
        env (Environment): the HMM environment
        actions (List[int]): a list of actions from time step 0 to t - 2
        observation (List[int]): a list of observations from time step 0 to t - 1 

    Returns:
        tuple: a tuple of 2 list, the first is a list of integer containing the most likely sequence of states, 
               the second is a list of probability of  max(m0:k). Both list should be of length t 

    """
    t = len(observations)
    if actions is None:
        actions = [None] * t

    # base case m_{0:0}=f_{0:0}
    raw_base_case = np.multiply(env.observation_dist(observations[0]), env.init_probs)
    s = sum(raw_base_case)
    base_case = raw_base_case / s

    # matrix formed by m_{0:k}
    m_mat = [base_case]
    # store states information at each time step
    states_info = []

    curr_m = base_case

    for k in range(1, t):
        next_m, state = viterbi_recurse(env, curr_m, actions[k - 1], observations[k])
        m_mat.append(next_m.copy())
        curr_m = next_m
        states_info.append(state.copy())

    last_prob = max(m_mat[t - 1])
    last_state = np.where(m_mat[t - 1] == last_prob)[0][0]
    res_state = [last_state]
    res_prob = [last_prob]

    for i in reversed(range(0, t - 1)):
        res_state.append(states_info[i][res_state[-1]])
        res_prob.append(m_mat[i][res_state[-1]])

    res_state.reverse()
    res_prob.reverse()
    return (res_state, res_prob)

