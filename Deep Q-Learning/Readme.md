
# Project Details

In this project, we train an agent to navigate (and collect bananas!) in a large, square world.

<img src="https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif" alt="Trained Agent" title="Trained Agent" style="max-width: 100%; align=left">

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

<li><strong><code>0</code></strong> - move forward.</li>
<li><strong><code>1</code></strong> - move backward.</li>
<li><strong><code>2</code></strong> - turn left.</li>
<li><strong><code>3</code></strong> - turn right.</li>

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

# Environment Setup

<p dir="auto">Download the environment from one of the links below.  You need only select the environment that matches your operating system:</p>
<ul dir="auto">
<li>Linux: <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip" rel="nofollow">click here</a></li>
<li>Mac OSX: <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip" rel="nofollow">click here</a></li>
<li>Windows (32-bit): <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip" rel="nofollow">click here</a></li>
<li>Windows (64-bit): <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip" rel="nofollow">click here</a></li>
</ul>
<p dir="auto">(<em>For Windows users</em>) Check out <a href="https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64" rel="nofollow">this link</a> if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.</p>
<p dir="auto">(<em>For AWS</em>) If you'd like to train the agent on AWS (and have not <a href="https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md">enabled a virtual screen</a>), then please use <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip" rel="nofollow">this link</a> to obtain the environment.</p>
</li>

<p dir="auto">Place the file in the course GitHub repository, in the <code>p1_navigation/</code> folder, and unzip (or decompress) the file.</p>
</li>
</ol>

# Instructions

Follow the instructions in <code>Navigation.ipynb</code> to implement and train the agent. The implementation uses Double Deep Q-Learning approach. I.e., there are two neural networks one for selecting the action and the other for evaluating the action.
The code has three important modules (classes): <code>QNetwork, Agent, and ReplayBuffer</code>. The <code>QNetwork</code> class contains the neural network structure using Pytorch. The <code>Agent</code> class contains necessary procedues to perform action selection and optimization. Finally, <code>ReplayBuffer</code> class contains procedure to store the transtions (state, next_state, reward, action, episode_termination_indicator) and sampling of such transtions.


```

```
