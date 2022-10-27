from random import gammavariate
import torch
from torch.optim import Adam
import numpy as np
import scipy
import time
from tqdm import tqdm

from buffer import Buffer
from model import Model

class Trainer:
    def __init__(self, env, timesteps_per_batch=2048, train_pi_iters=10, train_v_iters=10, 
                 pi_lr=3e-4, vf_lr=1e-3, gamma=0.99, lam=0.97, clip_ratio=0.2, device="cpu"):
        self.env = env
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.n

        self.obs_dim = (obs_dim[2], obs_dim[0], obs_dim[1])
        self.act_dim = (act_dim,)

        self.device = device

        self.epoch = 0

        self.timesteps_per_batch = timesteps_per_batch
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio

        self.buffer = Buffer(self.obs_dim, self.act_dim, self.timesteps_per_batch)

        self.model = Model(self.obs_dim, self.act_dim)
        self.pi_optimizer = Adam(self.model.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.model.vf.parameters(), lr=self.vf_lr)

    def train_one_epoch(self):

        ep_lens = []
        ep_rets = []

        ep_len = 0
        ep_ret = 0

        start = time.time()

        obs = self.env.reset()
        obs = np.moveaxis(obs, 2, 0)

        with torch.no_grad():
            for step in tqdm(range(self.timesteps_per_batch), leave=False):
                
                obs_tensor = self.np_to_device(obs)

                act, val, logp = self.model.step(obs_tensor)

                next_obs, rew, done, _ = self.env.step(act)
                
                self.buffer.store(obs, act, rew, val, logp)

                ep_len += 1
                ep_ret += rew

                if done:
                    ep_lens.append(ep_len)
                    ep_rets.append(ep_ret)
                    
                    self.buffer.finish_path(0)

                    obs = self.env.reset()
                    ep_len = 0
                    ep_ret = 0

                    done = False

                elif step == self.timesteps_per_batch-1:
                    val = self.model.critic(self.np_to_device(obs))
                    self.buffer.finish_path(val.numpy())
            
                next_obs = np.moveaxis(next_obs, 2, 0)
                obs = next_obs



        selfplay_time = time.time() - start
        start = time.time()

        data = self.buffer.get()

        pi_loss_old = self.get_pi_loss(data).item()
        v_loss_old = self.get_vf_loss(data).item()

        for i in tqdm(range(self.train_pi_iters), leave=False):
            self.pi_optimizer.zero_grad()
            pi_loss = self.get_pi_loss(data)
            pi_loss.backward()
            self.pi_optimizer.step()

        for i in tqdm(range(self.train_v_iters), leave=False):
            self.vf_optimizer.zero_grad()
            vf_loss = self.get_vf_loss(data)
            vf_loss.backward()
            self.vf_optimizer.step()

        training_time = time.time() - start

        self.epoch += 1

        return pi_loss_old, v_loss_old, ep_lens, ep_rets, selfplay_time, training_time


    def get_pi_loss(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        pi = self.model.actor_dist(obs)
        logp = pi.log_prob(act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        return loss_pi
    
    def get_vf_loss(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.model.critic(obs) - ret)**2).mean()

    def save_state(self):
        torch.save({
            "model": self.model.state_dict(),
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "vf_optimizer": self.vf_optimizer.state_dict(),
            "epoch": self.epoch,
        }, f"./results/weights/{self.epoch}.pt")

    def load_state(self, e):
        checkpoint = torch.load(f"./results/weights/{e}.pt")

        self.model.load_state_dict(checkpoint["model"])
        self.pi_optimizer.load_state_dict(checkpoint["pi_optimizer"])
        self.vf_optimizer.load_state_dict(checkpoint["vf_optimizer"])
        self.epoch = checkpoint["epoch"]

    def np_to_device(self, arr):
        return torch.as_tensor(arr, dtype=torch.float32).to(self.device)