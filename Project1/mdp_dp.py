### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
# Modified By Yanhua Li on 09/09/2022 for gym==0.25.2
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def find_valuefunction(P, value_function, state, action,gamma):
    val_function_inner = 0
    for trans_prob, next_state, reward_prob, _ in P[state][action]:
                    val_function_inner = val_function_inner + ( (trans_prob * (reward_prob + gamma * value_function[next_state]) 
                            ))
    return val_function_inner   

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state 
    """
    
    value_function = np.zeros(nS)
    
    
    while True: 
        delta = 0
        for i in range(nS):
            
            last_val_function = value_function[i]
            state = i
            current_policy =policy[i]

            val_function = 0           

            for j in range(nA):
                action = j
                """
                val_function_inner = 0

                for trans_prob, next_state, reward_prob, _ in P[state][action]:
                    val_function_inner = val_function_inner + ( (trans_prob * (reward_prob + gamma * value_function[next_state]) 
                            ))

                """
                val_function_inner = find_valuefunction(P, value_function, state, action, gamma)
                val_function =  val_function + (current_policy[action] *  val_function_inner)   
                ##print(val_function)   

            value_function[state] = val_function          
            

            delta = max(delta, abs(last_val_function - val_function))
            ##print(delta)
            
        if (delta < tol):
            break
            
                 
   
        


    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################


    ##print(value_function)
    return value_function 


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """
    
    new_policy = np.ones([nS, nA]) / nA # policy as a uniform distribution
    ##is_policy_stable = True
    for i in range(nS):
        state = i
        ##old_action = new_policy[state]

        """
        for j in range(nA):
                action = j
                
                val_function_inner = 0
                for trans_prob, next_state, reward_prob, _ in P[state][action]:
                    val_function_inner = val_function_inner + ( (trans_prob * (reward_prob + gamma * value_function[next_state]) 
                            ))
                val_function =  val_function + (current_policy[action] *  val_function_inner)   
                print(val_function)  
                
        """        
        max_vaue_index = np.argmax([find_valuefunction(P, value_from_policy, state, action, gamma) # list comprehension
                               for action in range(nA)])
        new_policy[state] = np.zeros (nA)
        new_policy[state,max_vaue_index] = 1
    


	############################
	# YOUR IMPLEMENTATION HERE #
    #                          #
	############################
    ##print(new_policy)
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
    while True:
        old_policy  = new_policy
        V = policy_evaluation(P, nS, nA, new_policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, V, gamma)
        if ((new_policy == old_policy).all()):
            break



	############################
	# YOUR IMPLEMENTATION HERE #
    #                          #
	############################
    return new_policy, V

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    V_new = V.copy()
    policy_new = np.zeros([nS, nA])

    while True:
        delta = 0
        for i in range(nS):
            state = i
            old_val_function = V_new[state] 
            val_function  = max([find_valuefunction(P, V_new, state, action, gamma) # list comprehension
                               for action in range(nA)])
            V_new[state] = val_function
            delta = max(delta, abs(old_val_function - val_function))

        if (delta < tol ):
            ## policy im
            # policy improvement function is used to extract the optimum policy found using above code when delat <0
            policy_new = policy_improvement(P, nS, nA, V_new , gamma)
            break


    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    return policy_new, V_new

def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob, _ = env.reset() # initialize the episode
        done = False
        action = np.argmax(policy[0])
        
        next_state, reward, done,info, _ = env.step(action) #takes action

        while not done:   # using "not truncated" as well, when using time_limited wrapper.
            
            if render:
                
                env.render() # render the game
                
            action = np.argmax(policy[next_state])
            next_state, reward, done,info, _ = env.step(action) #takes action
            

            ##print(reward)
            total_rewards = total_rewards + reward    
                   

            ############################
            # YOUR IMPLEMENTATION HERE #
            #                          #
            ############################
    ##print (total_rewards)  
    return total_rewards



