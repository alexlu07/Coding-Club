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
    def __init__(self, env, timesteps_per_batch=1024, train_pi_iters=2, train_v_iters=2, 
                 lr=1e-3, gamma=0.99, lam=0.97, clip_ratio=0.2, device="cpu"):
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
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio

        self.buffer = Buffer(self.obs_dim, self.act_dim, self.timesteps_per_batch)

        self.model = Model(self.obs_dim, self.act_dim)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

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

        loss_old = self.get_loss(data).item()

        for i in tqdm(range(self.train_pi_iters), leave=False):
            self.optimizer.zero_grad()
            loss = self.get_loss(data)
            loss.backward()
            self.optimizer.step()

        training_time = time.time() - start

        self.epoch += 1

        return loss_old, ep_lens, ep_rets, selfplay_time, training_time


    def get_loss(self, data):
        obs, act, adv, ret, logp_old = data['obs'], data['act'], data['adv'], data['ret'], data['logp']

        logp, val = self.model.chicken_nugget(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
    
        loss_vf = ((val - ret)**2).mean()

        loss = loss_pi + loss_vf
        return loss

    def save_state(self):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }, f"./results/weights/{self.epoch}.pt")

    def load_state(self, e):
        checkpoint = torch.load(f"./results/weights/{e}.pt")

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint["epoch"]

    def np_to_device(self, arr):
        return torch.as_tensor(arr, dtype=torch.float32).to(self.device)