import argparse

import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1'], default='CartPole-v1')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    
    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])


    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    iteration = 0
    training_history = []

    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()
        obs = preprocess(obs, env=args.env).unsqueeze(0)
        while not terminated:
            iteration += 1
            
            # TODO: Get action from DQN.
            action = dqn.act(obs, iteration).item()
            # Act in the true environment.
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Preprocess incoming observation.
            #if not terminated:
            next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
                
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!

            obs = torch.tensor(obs).to(device)
            next_obs = torch.tensor(next_obs).to(device)
            action = torch.tensor(action).unsqueeze(0).to(device)
            reward = torch.tensor(reward).unsqueeze(0).to(device)

            if terminated:
                terminated_bool = torch.tensor(1).unsqueeze(0)
            else:
                terminated_bool = torch.tensor(0).unsqueeze(0)

            memory.push(obs, action, next_obs, reward, terminated_bool)
            
            obs = next_obs

            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if episode % env_config["train_frequency"] == 0:       
                optimize(dqn, target_dqn, memory, optimizer=optimizer)
                
            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if episode % env_config['target_update_frequency'] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')
            training_history.append(mean_return)
            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                #torch.save(dqn, f'models/{args.env}_best.pt')
        
    # Close environment after training is completed.
    env.close()

plt.plot(training_history)
plt.grid()
plt.title('Training history - CartPole-v1')
plt.xlabel('Evaluation episode')
plt.ylabel('Mean return')
plt.ylim(0, 450)
plt.savefig('/Users/falk/Documents/python_projects/Reinforcement_learning/project/figures/CartPole_history_5.png',dpi=300)
plt.show()
