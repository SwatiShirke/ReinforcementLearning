#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        linear_input_size = 7 * 7 * 64
        self.FC_1 = nn.Linear(linear_input_size, 512)
        self.FC_2 = nn.Linear(512, num_actions)     
        #self.fc4 = nn.Linear(7 * 7 * 64, 512)
        #self.fc5 = nn.Linear(512, num_actions)   

    def forward(self, x):
        """
        Parameters:
        -----------
        x: input layer
        Returns:
        --------
        x: output layer
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))    
        x = F.relu(self.FC_1(x.reshape(x.size(0), -1)))
        x = self.FC_2(x)
        return x
    
        # x = F.relu(self.fc4(x.view(x.size(0), -1)))
        # return self.fc5(x)

