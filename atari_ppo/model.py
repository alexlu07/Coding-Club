import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions.categorical import Categorical

class Model(nn.Module):
    def __init__(self, obs_dim, act_dim, sample_obs):
        super().__init__()
    
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.make_networks(sample_obs)

    def make_networks(self, sample_obs):
        self.pi = mlp(self.obs_dim[0], [64, 64], self.act_dim[0], activation=nn.ReLU)
        self.vf = mlp(self.obs_dim[0], [64, 64], 1, activation=nn.ReLU)

    def step(self, obs, temp=1.0):
        with torch.no_grad():
            obs = self.initial_passthrough(obs)
            pi = self.pi_dist(obs, temp)
            a = pi.sample() 
            logp_a = pi.log_prob(a)
            v = self.vf(obs)

        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def pi_dist(self, obs, temp):
        dist = self.pi(obs)
        dist = (dist / temp).softmax(dim=-1)
        return Categorical(probs=dist)

    def chicken_nugget(self, obs, act):
        obs = self.initial_passthrough(obs)

        pi_dist = self.pi(obs)
        pi_dist = Categorical(logits=pi_dist)
        logp = pi_dist.log_prob(act)
        values = self.vf(obs)

        return logp, values.squeeze(), pi_dist.entropy()
        
    def critic(self, obs):
        return self.vf(self.initial_passthrough(obs))

    def initial_passthrough(self, obs):
        return obs

class ConvModel(Model):
    def make_networks(self, sample_obs):
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(-3),
        )

        # sample_obs = np.moveaxis(sample_obs, 2, 0)

        n_flatten = self.conv(torch.as_tensor(sample_obs, dtype=torch.float32)).shape[0]
        self.linear = nn.Linear(n_flatten, 512)

        self.pi = mlp(512, [64, 64], self.act_dim[0], activation=nn.ReLU)

        self.vf = mlp(512, [64, 64], 1, activation=nn.ReLU)

    def initial_passthrough(self, obs):
        x = self.conv(obs)
        x = F.relu(self.linear(x))

        return x        

def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ReLU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act(dim=-1) if act==torch.nn.Softmax else act()]
    return torch.nn.Sequential(*layers)
