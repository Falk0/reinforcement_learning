import argparse

import gymnasium as gym
import torch

import config_pong
from utils_pong import preprocess
from evaluate_pong import evaluate_policy
from dqn_pong import DQN, ReplayMemory, optimize
from gymnasium.wrappers import AtariPreprocessing

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'ALE/Pong-v5': config_pong.Pong
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
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


    for episode in range(env_config['n_episodes']):
        
        obs_list = []
        obs, info = env.reset()
        obs = preprocess(obs, env=args.env).unsqueeze(0)/255
        obs_list.append(obs)
        terminated = False
        
        
        for i in range(4):
            next_obs, reward, terminated, truncated, info = env.step(0)
            obs_list.append(preprocess(next_obs, env=args.env).unsqueeze(0)/255)
        
        obs_cat = torch.cat((obs_list[0], obs_list[1], obs_list[2], obs_list[3]), dim =0).unsqueeze(0)

        while not terminated:
            iteration += 1
            
            # TODO: Get action from DQN.
            action = dqn.act(obs_cat, iteration).item()
            # Act in the true environment.
            next_obs_list = []
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs_list.append(preprocess(next_obs, env=args.env).unsqueeze(0)/255)
            
            for i in range(4):
                next_obs, reward, terminated, truncated, info = env.step(0)
                next_obs_list.append(preprocess(next_obs, env=args.env).unsqueeze(0)/255)

     
            # Preprocess incoming observation.
            #if not terminated:
            next_obs_cat = torch.cat((next_obs_list[0], next_obs_list[1], next_obs_list[2], next_obs_list[3]), dim =0).unsqueeze(0)
            next_obs_cat = preprocess(next_obs_cat, env=args.env)

            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
    

            obs_cat = torch.tensor(obs_cat).to(device)
            next_obs_cat = torch.tensor(next_obs_cat).to(device)
            action = torch.tensor(action).unsqueeze(0).to(device)
            reward = torch.tensor(reward).unsqueeze(0).to(device)


            if terminated:
                terminated_bool = torch.tensor(1).unsqueeze(0)
            else:
                terminated_bool = torch.tensor(0).unsqueeze(0)

            memory.push(obs_cat, action, next_obs_cat, reward, terminated_bool)
            
            obs_cat = next_obs_cat

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

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                #torch.save(dqn, f'models/{args.env}_best.pt')
        
    # Close environment after training is completed.
    env.close()
