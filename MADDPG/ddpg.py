from network import Actor_QNetwork, Critic_QNetwork
import torch.optim as optim
import torch
import numpy as np
import random
from utilities import soft_update, transpose_to_tensor, transpose_list, hard_update
from noise import OUNoise, BetaNoise, GaussNoise, WeightedNoise

class DDPGAgent:
    def __init__(self, state_size, action_size, fc1_node, fc2_node, num_atomsp, lr_actor, lr_critic, l2_decay, noise_typep, OU_mu, OU_theta, OU_sigma, device, Num_Agent, desired_distance, scalarp, scalar_decay, Batch_Nor = True, Q_Distribution=False, log_active=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = device 
        self.noise_type = noise_typep
        self.num_atoms = num_atomsp
        
                       
        # parameter noise
        self.distances = []
        self.desired_distance = desired_distance
        self.scalar = scalarp
        self.scalar_decay = scalar_decay

        # Actor Network (w/ Target Network)
        self.actor_local     = Actor_QNetwork(state_size, action_size, fc1_node, fc2_node, Batch_Nor)
        self.actor_target    = Actor_QNetwork(state_size, action_size, fc1_node, fc2_node, Batch_Nor)
        
        hard_update(self.actor_target,self.actor_local)  
        
        self.actor_local.to(self.device)
        self.actor_target.to(self.device)        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor, weight_decay=l2_decay)
        
        self.actor_noised = Actor_QNetwork(state_size, action_size, fc1_node, fc2_node, Batch_Nor).to(device)

        # Critic Network (w/ Target Network)
        """Double Q-Learning uses Q-network-1 (qnetwork_local) to select actions and Q-network-2 (qnetwork_target) to evaluate the selected actions."""
        self.critic_local  = Critic_QNetwork(state_size, Num_Agent*action_size, fc1_node, fc2_node, self.num_atoms, Batch_Nor, Q_Distribution, log_active)
        self.critic_target = Critic_QNetwork(state_size, Num_Agent*action_size, fc1_node, fc2_node, self.num_atoms, Batch_Nor, Q_Distribution, log_active)
        
        hard_update(self.critic_target,self.critic_local)  
        self.critic_local.to(self.device)
        self.critic_target.to(self.device)         
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=l2_decay)

        # Noise process
        if self.noise_type=='OUNoise': # if we're using OUNoise it needs to be initialised as it is an autocorrelated process
            self.noise = OUNoise(action_size, OU_mu, OU_theta, OU_sigma)

    def act(self, state, noise_scale):
        state = (torch.from_numpy(state)).float().to(self.device) 
        self.actor_local.train(mode=False)
        self.actor_noised.train(mode=False)
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        
        if self.noise_type == 'param':
            if self.noise_type == "param":
                # hard copy the actor_regular to actor_noised
                self.actor_noised.load_state_dict(self.actor_local.state_dict().copy())
                # add noise to the copy
                self.actor_noised.add_parameter_noise(self.scalar)
                # get the next action values from the noised actor
                action_noised = self.actor_noised(state).cpu().data.numpy()
                # meassure the distance between the action values from the regular and 
                # the noised actor to adjust the amount of noise that will be added next round
                distance = np.sqrt(np.mean(np.square(action-action_noised)))
                # for stats and print only
                self.distances.append(distance)
                # adjust the amount of noise given to the actor_noised
                if distance > self.desired_distance:
                    self.scalar *= self.scalar_decay
                if distance < self.desired_distance:
                    self.scalar /= self.scalar_decay
                # set the noised action as action
                action = action_noised
        elif self.noise_type=='OUNoise':
            noise = self.noise.noise()
            action += noise_scale*(noise)            
        elif self.noise_type=='BetaNoise':
            action = BetaNoise(action, noise_scale)
        elif self.noise_type=='GaussNoise':
            action = GaussNoise(action, noise_scale)
        elif self.noise_type=='WeightedNoise':
            action = WeightedNoise(action, noise_scale)
        
        self.actor_local.train(mode=True)    
        return np.clip(action, -1, 1)
    
    
    