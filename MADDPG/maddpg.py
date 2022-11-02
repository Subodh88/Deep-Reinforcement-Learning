from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list, hard_update
import numpy as np
import torch.nn.functional as F



class MADDPG:
    def __init__(self, p):

        super(MADDPG, self).__init__()
        
        self.maddpg_agent = [DDPGAgent(p['state_size'], p['action_size'], p['hidden_in_size'], p['hidden_out_size'], p['num_atoms'], p['lr_actor'], p['lr_critic'],p['l2_decay'],p['noise_type'],p['OU_mu'],p['OU_theta'],p['OU_sigma'],p['device'],p['Total_Agent'],p['desired_distance'],p['Param_scale_Noise'],p['Param_scale_Noise_decay'],p['Batch_Normalize'],p['Q_Distribution'],p['QD_log_active']),
        DDPGAgent(p['state_size'], p['action_size'], p['hidden_in_size'], p['hidden_out_size'], p['num_atoms'], p['lr_actor'], p['lr_critic'],p['l2_decay'],p['noise_type'],p['OU_mu'],p['OU_theta'],p['OU_sigma'],p['device'],p['Total_Agent'],p['desired_distance'],p['Param_scale_Noise'],p['Param_scale_Noise_decay'],p['Batch_Normalize'],p['Q_Distribution'],p['QD_log_active'])]
                             
        
        self.discount_rate = p['discount_rate']
        self.tau = p['tau']
        self.n_steps = p['n_steps']
        self.num_atoms = p['num_atoms']
        self.vmin = p['vmin']
        self.vmax = p['vmax']
        self.device = p['device']
        self.Tagent = p['Total_Agent']
        self.QD_Active = p['Q_Distribution'] 
        self.Mnoise_type = p['noise_type']
        self._batch_size = p['batchsize']
       
        self.iter = 0
        
        
    def act(self, obs_all_agents, noise_scale=np.zeros(2)):
        result = []
        for i in range(self.Tagent):
            result.append(self.maddpg_agent[i].act(obs_all_agents[i:i+1], noise_scale[i]))

        result = np.concatenate(result)
        return result   
    

    def update_targets(self,agent_num):
        """soft update targets"""
        self.iter += 1
        ddpg_agent = self.maddpg_agent[agent_num]
        soft_update(ddpg_agent.actor_target,  ddpg_agent.actor_local,  self.tau)
        soft_update(ddpg_agent.critic_target, ddpg_agent.critic_local, self.tau)
        
    def hard_update_targets(self,agent_num):
        """soft update targets"""
        self.iter += 1
        ddpg_agent = self.maddpg_agent[agent_num]
        hard_update(ddpg_agent.actor_target,  ddpg_agent.actor_local)
        hard_update(ddpg_agent.critic_target, ddpg_agent.critic_local)
        
    def save_models(self):
        for i in range(0,self.Tagent,1):
            icurr = i+1 
            self.actor_name  = str(icurr) + 'Actor_Local_'  + self.Mnoise_type + '.pth'
            self.critic_name = str(icurr) + 'Critic_Local_' + self.Mnoise_type + '.pth'

            agent = self.maddpg_agent[i]
            torch.save(agent.actor_local.state_dict(),  self.actor_name)
            torch.save(agent.critic_local.state_dict(), self.critic_name)

    def load_models(self):
        for i in range(0,self.Tagent,1):
            icurr = i+1 
            self.actor_name  = str(icurr) + 'Actor_Local_'  + self.Mnoise_type + '.pth'
            self.critic_name = str(icurr) + 'Critic_Local_' + self.Mnoise_type + '.pth'

            agent = self.maddpg_agent[i]
        
            agent.actor_local.load_state_dict(torch.load(self.actor_name))
            agent.critic_local.load_state_dict(torch.load(self.critic_name))
            agent.actor_target.load_state_dict(torch.load(self.actor_name))
            agent.critic_target.load_state_dict(torch.load(self.critic_name))
        print('Models loaded succesfully')

    def update(self, experiences, agent_number):

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(self.Tagent):
            states.append(torch.from_numpy(np.vstack([e.state[i] for e in experiences if e is not None])).float().to(self.device))
            actions.append(torch.from_numpy(np.vstack([e.action[i] for e in experiences if e is not None])).float().to(self.device))
            rewards.append(torch.from_numpy(np.vstack([e.reward[i] for e in experiences if e is not None])).float().to(self.device))
            next_states.append(torch.from_numpy(np.vstack([e.next_state[i] for e in experiences if e is not None])).float().to(self.device))
            dones.append(torch.from_numpy(np.vstack([e.done[i] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device))

        state = states[agent_number]
        action = actions[agent_number]
        reward = rewards[agent_number]
        next_state = next_states[agent_number]
        done = dones[agent_number]

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = [self.maddpg_agent[i].actor_target(states[i]) for i in range(self.Tagent)]
        actions_next = torch.cat(actions_next, dim=1)
        Q_targets_next = self.maddpg_agent[agent_number].critic_target(next_state, actions_next)
        Q_targets_next = Q_targets_next.detach()
        # Compute Q targets for current states (y_i)
        Q_targets = reward + (self.discount_rate * Q_targets_next * (1 - done))
        # Compute critic loss
        actions_exp = torch.cat(actions, dim=1)
        Q_expected = self.maddpg_agent[agent_number].critic_local(state, actions_exp)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.maddpg_agent[agent_number].critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.maddpg_agent[agent_number].critic_local.parameters(), 1)
        self.maddpg_agent[agent_number].critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = [ self.maddpg_agent[i].actor_local(state) if i == agent_number \
                   else self.maddpg_agent[i].actor_local(state).detach()
                   for i, state in enumerate(states) ]
        actions_pred = torch.cat(actions_pred, dim=1)

        actor_loss = -self.maddpg_agent[agent_number].critic_local(state, actions_pred).mean()
        # Minimize the loss
        self.maddpg_agent[agent_number].actor_optimizer.zero_grad()
        actor_loss.backward()
        self.maddpg_agent[agent_number].actor_optimizer.step()
        
                
        
    def to_categorical(self, rewards, probs, dones):
        """
        Credit to Matt Doll and Shangtong Zhang for this function:
        https://github.com/whiterabbitobj
        https://github.com/ShangtongZhang
        """

        # Create local vars to keep code more concise
        vmin = self.vmin
        vmax = self.vmax
        atoms = self.atoms
        num_atoms = self.num_atoms
        n_steps = self.n_steps
        discount_rate = self.discount_rate
        
        # this is the increment between atoms
        delta_z = (vmax - vmin) / (num_atoms - 1)

        # projecting the rewards to the atoms
        projected_atoms = rewards + discount_rate**n_steps * atoms * (1 - dones)
        projected_atoms.clamp_(vmin, vmax) # vmin/vmax are arbitary so any observations falling outside this range will be cliped
        b = (projected_atoms - vmin) / delta_z

        # precision is reduced to prevent e.g 99.00000...ceil() evaluating to 100 instead of 99.
        precision = 1
        b = torch.round(b * 10**precision) / 10**precision
        lower_bound = b.floor()
        upper_bound = b.ceil()

        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs

        # initialising projected_probs
        projected_probs = torch.tensor(np.zeros(probs.size())).to(device)

        # a bit like one-hot encoding but for the specified atoms
        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())
        return projected_probs.float()
