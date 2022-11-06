import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical

class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pi = nn.Sequential(
            nn.Conv2d(128, 4,  1),
            nn.ReLU(),
            nn.Flatten(-3),
            nn.Linear(4*obs_dim[1]*obs_dim[2], act_dim[0])
        )

        self.vf = nn.Sequential(
            nn.Conv2d(128, 2, 1),
            nn.ReLU(),
            nn.Flatten(-3),
            mlp(2*obs_dim[1]*obs_dim[2], [64], 1)
        )

    def step(self, obs):
        with torch.no_grad():
            obs = self.initial_passthrough(obs)
            pi = self.pi_dist(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a)
            v = self.vf(obs)

        return a.numpy(), v.numpy(), logp_a.numpy()

    def pi_dist(self, obs):
        dist = self.pi(obs)
        return Categorical(logits=dist)

    def chicken_nugget(self, obs, act):
        obs = self.initial_passthrough(obs)

        pi_dist = self.pi(obs)
        pi_dist = Categorical(logits=pi_dist)
        logp = pi_dist.log_prob(act)
        values = self.vf(obs)

        return logp, values

    def critic(self, obs):
        return self.vf(self.initial_passthrough(obs))

    def initial_passthrough(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

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
