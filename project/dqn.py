import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward, terminated):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward, terminated)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, observation, iteration, exploit=False):
        
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.
        action_tensor = self.forward(observation)

        epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1.0 * iteration / self.anneal_length)
  
        if np.random.rand() > epsilon or exploit == True: 
            action = torch.argmax(action_tensor, axis=1)
 
        else: 
            action = np.random.choice(self.n_actions) 
            action = torch.tensor(action)
            #print('explore')
        
        return action
        #raise NotImplmentedError

def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    
    sample  = memory.sample(dqn.batch_size)
    
    obs, act, next_obs, reward, terminated = sample
    

    concatenated_terminated = torch.cat(terminated, dim=0)
    concatenated_terminated = concatenated_terminated.to(device)
    concatenated_obs = torch.cat(obs, dim=0)
    concatenated_obs = concatenated_obs.to(device)
    concatenated_act = torch.cat(act, dim=0)
    concatenated_act = concatenated_act.to(device)
    concatenated_next_obs = torch.cat(next_obs, dim=0)
    concatenated_next_obs = concatenated_next_obs.to(device)
    concatenated_reward = torch.cat(reward, dim=0)
    concatenated_reward = concatenated_reward.to(device)

    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    
    
    q_values = dqn.forward(concatenated_obs)
    Q_values = torch.gather(q_values, dim=1, index=concatenated_act.unsqueeze(1))


    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    
    # Get the Q-values for the next states from the target DQN
    next_q_values = target_dqn(concatenated_next_obs)

    # Select the maximum Q-value for each next state
    max_next_q_values, _ = torch.max(next_q_values, dim=1)
    
    # Compute the Q-value targets for non-terminal transitions

    Q_targets = concatenated_reward + (dqn.gamma * max_next_q_values * (1 - concatenated_terminated))
    Q_targets = Q_targets.unsqueeze(1)
    
    # We need to detach Q_targets from the computation graph to avoid backpropagation issues
    Q_target = Q_targets.detach()
    
    # Compute loss. 
    loss = F.mse_loss(Q_values, Q_target)

    # Perform gradient descent.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    return loss.item()
