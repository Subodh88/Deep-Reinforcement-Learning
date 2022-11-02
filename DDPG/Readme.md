# Environment

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints bound between a value of -1 and 1.

The environemnt is considered solved, when an average reward of 30 points is achieved over a consecutive 100 episodes.

# Seting up the environment

**1. One-agent version**

Download the environment from one of the links below.  You need only select the environment that matches your operating system:

* Linux:[click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX:[click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit):[click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit):[click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

**2. Twenty-agent version**

Download the environment from one of the links below.  You need only select the environment that matches your operating system:

* Linux:[click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* Mac OSX:[click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit):[click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* Windows (64-bit):[click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

# Instructions

Follow the instructions in `DDPG_Parameter_Noise.ipynb` to implement and train the agent. The implementation uses DDPG algorithm.
The code has three important modules (classes): `Actor_QNetwork, Critic_QNetwork, Agent, and ReplayBuffer` .
The `Actor_QNetwork` class contains the neural network structure using Pytorch for the actor.
The `Critic_QNetwork` class contains the neural network structure using Pytorch for the critic.
The `Agent` class contains necessary procedues to perform action selection and optimization.
Finally, `ReplayBuffer` class contains procedure to store the transtions (state, next_state, reward, action, episode_termination_indicator) and sampling of such transtions.
See the variables below for flexible setup

Random_Seed           = 5                              "Starting random seed"

hidden_layer_1        = 300                          "# of nodes in layer 1"

hidden_layer_2        = 400                          ""# of nodes in layer 1"

Normal_noise          = "Normal"                    "Standard normal draw for adding noise"

OU_noise              = "ou"                                   "Ornstein-Unlenbeck process for adding noise"

Param_noise           = "param"                        "Adaptive parameter noise (APN)"

curr_noise            = Param_noise                    "Select the noise type to be used in training"

desired_distance_pass = .5                    "Tolerance distance between action and nosiy action value. If distance (1/[action-noisy_action]^2) is greater than                                                                   the tolerance, then decrease the pertubation (scaler_pass) using exponential decay                                                                   (scaler_pass = scaler_pass*scalar_decay_pass) "

scaler_pass           = .05                                    "parameter to limit the perturbation in APN. A value of .05 indicates that noisy action will be prodeuced by changing                                                                   actor's parameter value in a range of +5% to -5%"

scalar_decay_pass     = .99

normal_scalar_pass    = .25

Load_Models   = 1                                    "Set to 1 to use previously trained model as the starting value"
