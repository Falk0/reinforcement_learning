import torch

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")

def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v1']:
        return torch.tensor(obs, device=device).float()
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')
