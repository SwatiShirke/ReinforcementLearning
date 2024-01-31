#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:11:22 2019

@author: huiminren
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
"""
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.

    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.
'''
#-------------------------------------------------------------------------



def initial_policy(observation):

    """A policy that sticks if the player score is >= 20 and hit otherwise

    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    if(observation[0]>=20):
        action = 0
    else:
        action = 1

    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    return action


def generate_episode(policy, env):

    state_list = []
    reward_list = []
    action_list = []
    tuple_list = []
    ob, _ = env.reset() # initialize the episode
    done = False
    ##action = np.argmax(policy(ob))
    ##next_state, reward, done,info, _ = env.step(action)
    next_state = ob

    while not done:
        state_list.append(next_state)
        current_state = next_state
        action = policy(next_state)          
        next_state, reward, done,info, _ = env.step(action)
        action_list.append(action)
        reward_list.append(reward)
        tuple_list.append( (current_state,action, reward))

    ##print(state_list)
    return state_list,action_list,reward_list,tuple_list

def generate_episode_control(policy, Q, nA, epsilon, env):

    state_list = []
    reward_list = []
    action_list = []
    tuple_list = []
    ob, _ = env.reset() # initialize the episode
    done = False
    ##action = np.argmax(policy(ob))
    ##next_state, reward, done,info, _ = env.step(action)
    next_state = ob

    while not done:
        state_list.append(next_state)
        current_state = next_state
        action = policy(Q, next_state,nA, epsilon)          
        next_state, reward, done,info, _ = env.step(action)
        action_list.append(action)
        reward_list.append(reward)
        tuple_list.append( (current_state,action, reward))

    ##print(state_list)
    return state_list,action_list,reward_list,tuple_list


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float) 
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################


    for episode in range(n_episodes):
        #follow the policy to collect the episode 
        states,actions,rewards,tuple_list = generate_episode(policy, env)
        total_returns = 0
        ##print(len(tuple_list))
        for t in range(len(states)-1,-1,-1):
            Reward = rewards[t]
            current_state = states[t]

            total_returns = gamma* total_returns +Reward
            if current_state not in states[: t]:
                returns_count[current_state] += 1 
                returns_sum[current_state] += total_returns
                ##V[current_state] += ( total_returns - V[current_state]) / returns_count[current_state] 
                V[current_state] = returns_sum[current_state] / returns_count[current_state]
                

    ##print (V)        
    return V


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: 
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 - epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """

    p = np.random.random()
    if p <= epsilon:
        ##action = Q["lambda"]
        action = random.choice(range(nA))
    else: 
        action = np.argmax(Q[state])
        
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    return action


def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-0.1/n_episode during each episode
    and episode must > 0.
    """

    ##returns_sum = defaultdict(float)
    ##returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    returns_count = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    nA = env.action_space.n

    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    policy = epsilon_greedy
    for episode in range(n_episodes):
        #follow the policy to collect the episode 
        states,actions,rewards,tuple_list = generate_episode_control(policy, Q, nA, epsilon, env )
        total_returns = 0
        visited_list = []
        for t in range(len(tuple_list)-1,-1,-1):
            current_state,action,Reward = tuple_list[t]
            #current_state = tuple[0]
            #Reward = tuple(2)


            total_returns = gamma* total_returns +Reward
            if  ( current_state, action) not in visited_list:
                ##print (action)
                ##action = int(action)
                ##print(current_state)
                returns_count[current_state][action] += 1 
                returns_sum[current_state][action] += total_returns
                ##Q[current_state][action] += ( total_returns - Q[current_state][action]) / returns_count[current_state] 
                Q[current_state][action] = ( returns_sum[current_state][action] ) / returns_count[current_state][action] 
                
                visited_list.append((current_state,action))
                


    return Q
