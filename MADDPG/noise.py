import numpy as np
import torch

class OUNoise:

    def __init__(self, action_size, mu, theta, sigma):
        self.action_size = action_size
        self.mu = mu                                                # mu - the long-term mean
        self.theta = theta                                          # theta - the mean reversion strength
        self.sigma = sigma                                          # sigma - the noise magnitude
        self.state = np.ones(self.action_size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_size) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x)) # mean reversion plus noise
        self.state = x + dx
        return self.state
        
    
def BetaNoise(action, noise_scale):    
        
    sign = np.sign(action)                              # tracking the sign so we can flip the samples later
    action = abs(action)                                # we only use right tail of beta
    alpha = 1/noise_scale                               # this determines the how contentrated the beta dsn is
    value = 0.5+action/2                                # converting from action space of -1 to 1 to beta space of 0 to 1
    beta = alpha*(1-value)/value                        # calculating beta
    beta = beta + 1.0*((alpha-beta)/alpha)              # adding a little bit to beta prevents actions getting stuck at -1 or 1
    sample = np.random.beta(alpha, beta)                # sampling from the beta distribution
    sample = sign*sample+(1-sign)/2                     # flipping sample if sign is <0 since we only use right tail of beta dsn
                    
    action_output = 2*sample-1                          # converting back to action space -1 to 1
    return action_output              

def GaussNoise(action, noise_scale):

    n = np.random.normal(0, 1, len(action))                                    # create some standard normal noise
    return (action+(noise_scale*n)) 

def WeightedNoise(action, noise_scale):
    """
    Returns the epsilon scaled noise distribution for adding to Actor
    calculated action policy.
    """
    target = np.random.uniform(-1,1,2)     # the action space is -1 to 1
    action = noise_scale*target+(1-noise_scale)*action      # take a weighted average with noise_scale as the noise weight
    return (action)