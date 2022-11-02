from collections import namedtuple, deque
import random
import numpy as np

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size,  n_steps, discount_rate, Nstep_Exp = False):
        """Initialize a ReplayBuffer object."""
        self._Nstep_Exp    = Nstep_Exp
        self.max_size      = buffer_size
        self.n_steps       = n_steps
        self.discount_rate = discount_rate
        self.memory        = deque(maxlen=self.max_size) 
        self.n_step_deque  = deque(maxlen=self.n_steps)              # new experience first goes here until n timesteps have passed  
        self.experience    = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if(self._Nstep_Exp):
            e = self.experience(state, action, reward, next_state, done)
            self.n_step_deque.append(e)
            if len(self.n_step_deque)==self.n_steps:             # once the deque has n steps we can start accumilating rewards
                start_obs, start_action, start_reward, start_next_obs, start_done = self.n_step_deque[0]  # first experience
                n_obs, n_action, n_reward, n_next_obs, n_done = self.n_step_deque[-1] # last experience
                
                summed_reward = np.zeros(2)                                    # initialise
                for i,n_transition in enumerate(self.n_step_deque):            # for each experience
                    obs, action, reward, next_obs, done = n_transition         
                    summed_reward += reward*self.discount_rate**(i+1)          # accumulate rewards
                    if np.any(done):                                           # stop if done
                        break
                """we take first obs and action, summed rewards and last next obs and done"""     
                e = self.experience(start_obs, start_action, summed_reward, n_next_obs, n_done)                         
        else:
            e = self.experience(state, action, reward, next_state, done)            
        self.memory.append(e)
    
    def sample(self,batch_size):
        """Randomly sample a batch of experiences from memory. Returns a list of length batch_size"""
        experiences = random.sample(self.memory, k=batch_size)
        return experiences    
  
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def reset(self):
        self.n_step_deque = deque(maxlen=self.n_steps) # reset n-step deque between episodes