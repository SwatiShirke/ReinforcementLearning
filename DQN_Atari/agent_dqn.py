#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from agent import Agent
from dqn_model import DQN
import math

import shutil
import wandb 
import numpy as np
HOME_PATH = "/home/svshirke/" ##define path to save model
output_path = "/home/svshirke/RLdqn/Project3/out/"

job_ID =  os.environ.get('SLURM_JOB_ID')       #os.environ.get('SLURM_JOB_ID')
job_location = output_path + f"{job_ID}/"
print (job_location)
if os.path.exists(job_location):
    shutil.rmtree(job_location)
    print(f"deleted previous folder from {job_location}")
os.mkdir(job_location)

"""
you can import any package and define any extra function as you need
"""
torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.learning_rate = 1e-4 #how fast network paramters are updated
        self.target_update_freq = 5000
        self.learning_starts = 5000


        self.in_channels = 4 # since the input is [84, 84, 4]
        self.num_actions = env.action_space.n # output size
        self.gamma = 0.99 #weight for future rewards
        self.replay_buffer_size = 10000 #size of replay buffer.. max no of tuples it can hold 
        self.batch_size = 32 # no of tuples to be poped out from the buffer at a time
        
        self.epsilon_step = 1000000
        self.initial_epsilon = 1.0 #init value for epsilon - used for epsilon -greedy policy
        self.final_epsilon = 0.025
        self.epsilon_decay  =  (self.initial_epsilon - self.final_epsilon)/ self.epsilon_step
        self.epsilon = self.initial_epsilon
        self.nA = 4  #no of actions 
        self.TAU = 0.005
        self.count = 0   
        self.num_episodes = 50000      
        self.loss = 0  
        self.learning_start = 5000 #start learning after these 
        self.step_count =0 ## no of steps executed by the agent so far 
        self.target_update_steps = 5000 #update target after these many steps
        self.n_observations = (84,84,4) ##state dimentions 

        ##init GPU/CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.Transition = namedtuple('Transition',
        #             ('state', 'action', 'next_state', 'reward','done_flag'))
        ##create q function and target q net
        self.policy_net = DQN(self.in_channels, self.nA).to(self.device)
        self.target_net = DQN(self.in_channels, self.nA).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        #self.target_net.eval()       
        
        
        #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate,  amsgrad=True) #init optimizer
        self.optimizer = optim.Adam(params = self.policy_net.parameters(), lr = self.learning_rate, 
                                   betas=(0.5, 0.999))
        self.loss_function = nn.SmoothL1Loss() #initialize huber loss
        self.clip_grad_value = 100
        self.Buffer = deque([],maxlen=self.replay_buffer_size) ##initialize buffer
           

        ##logger paramters and init logger to debug
        
        wandb.init(
            project="BreakOut_latest",

            config={
            "learning_rate":self.learning_rate, 
            "replay_buffer_size":self.replay_buffer_size, 
            "batch_size":self.batch_size,
            "gamma":self.gamma,
            "epsilon_decay":self.epsilon_decay, 
            "initial_epsilon":self.initial_epsilon, 
            "final_epsilon":self.final_epsilon,
            "self.epsilon": self.epsilon,
            "nA":self.nA,
            "TAU":self.TAU, 
            "count":self.count,
            "loss":self.loss,   
            "num_episodes":self.num_episodes,  

            }
            )
  
        
        

        

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            #model_path = "/home/svshirke/RLdqn/Project3/out/317416/00017000.pth" 
            self.model = torch.load('/home/svshirke/RLdqn/Project3/out/317416/00022000.pth')
            #
            self.policy_net.load_state_dict(self.model)
            self.policy_net.eval()

            #self.checkpoint = torch.load('/home/svshirke/RLdqn/Project3/out/317416/00022000.pth',map_location=self.device)
            # self.checkpoint = torch.load(self.path+'Palawat_PER5000_DBDQN.pth',map_location=device)

            #self.policy_net = self.checkpoint[DQN(self.in_channels, self.nA).to(self.device)]
            #self.policy_net.load_state_dict(self.checkpoint['state_dict'])
            #for parameter in self.policy_net.parameters():
            #    parameter.requires_grad = False
            #self.policy_net.eval()
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        if test:
            with torch.no_grad():
                    state = observation.transpose(2,0,1)
                    state = (torch.tensor(state).unsqueeze(0)/255.0).to(self.device).float()
                    action_values = self.policy_net(state).detach().cpu().numpy()
                    action = np.argmax(action_values[0])

        else:
            sample = random.random()   
            if self.step_count < self.learning_starts:   
                action = random.randrange(self.nA)
            elif sample <= self.epsilon:            
                action = random.randrange(self.nA)

            else:
                with torch.no_grad():
                    state = observation.transpose(2,0,1)
                    state = (torch.tensor(state).unsqueeze(0)/255.0).to(self.device).float()
                    action_values = self.policy_net(state).detach().cpu().numpy()
                    action = np.argmax(action_values[0])

                    #state_tensor = torch.tensor(observation, dtype=torch.float).to(self.device).unsqueeze(0)
                    #state_tensor = state_tensor.permute([0,3,1,2])
                    #action_values = self.policy_net(state_tensor).cpu().numpy()
                    #print(action_values)
                    #action = np.argmax(action_values[0])
                    #print(action)
                    #print(action)
                    #increase step count 
                    self.step_count += 1       
                
        ##decrease epsilon 
        if self.epsilon > self.final_epsilon:
                self.epsilon -= self.epsilon_decay        
        #print(action)
        return action
    
    

    def push(self,state, action, next_state, reward, done_flag):
        self.Buffer.append((state, action, next_state, reward, done_flag))
        
        
    def replay_buffer(self):
        batch = random.sample(self.Buffer, min(len(self.Buffer), self.batch_size))    
        batch = zip(*batch)
        return batch

    
        
    
    def train(self):
        """
        Implement your training algorithm here
        """
        if torch.cuda.is_available():
            num_episodes = self.num_episodes
        else:
            num_episodes = 50
        #reward_array = []
        
        for i_episode in range(self.num_episodes):
            terminated = False
            truncated = False
            state = self.env.reset() # shape is [84,84,4] 4 images 
            total_reward = 0
            while not terminated and not truncated:
                action = self.make_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward = total_reward + reward  
                done_flag = 0 
                if terminated or truncated:
                    done_flag = 1         
                self.push(state, action, next_state, reward, done_flag)

                state = next_state     #Move to the next state
                self.train_inner()      #Perform one step of the optimization (on the policy network)
                                
                if self.step_count%self.target_update_freq == 0: ##fixed target is updated after 5000 setps 
                    self.target_net.load_state_dict(self.policy_net.state_dict())             
    
                if done_flag == 1:
                    break
            
            
            """
            if len(reward_array)<30:
                average_reward = sum(reward_array[-len(reward_array):])/len(reward_array)
                #print("Pt-6")
            else:
                average_reward = sum(reward_array[-30:])/30
                #print("Pt-7")
            """
            #print(total_reward)
            wandb.log({"reward_episode": total_reward})
            wandb.log({"self.epsilon": self.epsilon})
            wandb.log({"loss": self.loss})
            

            """
            ##decrease epsilon 
            if self.epsilon > self.final_epsilon:
                self.epsilon -= ((self.initial_epsilon - self.final_epsilon)/ self.epsilon_decay )
            else: 
                self.epsilon = self.final_epsilon
            """
            ##print(average_reward)

            """
            if i_episode%30==0:
                total_reward /= 30 
                print(total_reward)
                ##print(total_reward)
                wandb.log({"reward_episode": total_reward})
                total_reward = 0

            """
            if i_episode%1000 ==0 :
                model_path = job_location + f"{i_episode:08d}.pth"
                torch.save(self.policy_net.state_dict(), model_path)
        
    
             
        
        
    def train_inner(self):
        
        if self.step_count < self.learning_starts:
            return

        """
        state_batch, action_batch, next_state_batch,reward_batch, done_batch   = self.replay_buffer()

        state_batch = torch.tensor(np.array(state_batch)).to(self.device).float()
        action_batch = torch.tensor(action_batch).to(self.device)
        reward_batch = torch.tensor(np.array(reward_batch)).to(self.device).float()
        done_batch = torch.tensor(np.array(done_batch)).to(self.device).float()
        next_state_batch = torch.tensor(np.array(next_state_batch)).to(self.device).float()
        
        next_state_batch = next_state_batch.permute([0,3,1,2])
        state_batch = state_batch.permute([0, 3, 1, 2])
        
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            #indices = torch.argmax(next_q_values).unsqueeze(1)
            max_next_q_values = next_q_values.max(1)[0]
            

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
        expected_state_action_values = ((1-  done_batch )* self.gamma * max_next_q_values) + reward_batch
        self.loss = self.loss_function(state_action_values, expected_state_action_values)      
        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.clip_grad_value)
        self.optimizer.step()
        """
        states_bt, actions_bt, next_states_bt, rewards_bt, done_batch = self.replay_buffer()
        # convert numpy to torch.tensor
        #states_bt = (torch.tensor(states_bt).type(dtype)/ 255.0 ).to(device)
        states_bt = (torch.tensor(np.array(states_bt))).to(self.device).float()
        states_bt = states_bt.permute([0,3,1,2])
        states_bt = states_bt /255.0
        actions_bt = (torch.tensor(actions_bt).long()).to(self.device)
        next_states_bt = (torch.tensor(np.array(next_states_bt))).to(self.device).float()
        next_states_bt = next_states_bt.permute([0,3,1,2])
        next_states_bt = next_states_bt / 255.0

        rewards_bt = (torch.tensor(np.array(rewards_bt))).to(self.device).float()
        #dones_bt = (torch.tensor(dones_bt).type(dtype)).to(device)
        done_batch = torch.tensor(np.array(done_batch)).to(self.device).float()
        
        
        
        # compute current Q value, self.Q() takes state and output value 
        # for each state-action pair
        # self.Q(state).gather(dim,index) helps us choose Q based on action taken.
        current_Q_values = self.policy_net(states_bt).gather(1,actions_bt.unsqueeze(1)).squeeze()
        # compute next Q value based on which action gives max Q values
        # detach next Q value since we don't want gradients for next Q to propagated
        # if done, next_Q_values = 0
        next_Q_values = (1-done_batch) * self.target_net(next_states_bt).detach().max(1)[0]
        # compute the expected Q value
        expected_Q_values = rewards_bt + self.gamma * next_Q_values
        # compute loss
#                 loss = F.MSELoss(current_Q_values, expected_Q_values.unsqueeze(1))
        loss = F.smooth_l1_loss(current_Q_values, expected_Q_values)
        self.loss = loss
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # why clip gradients to (-1,1)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # periodically update the target netowrk by Q network to target Q network
        

        
        









