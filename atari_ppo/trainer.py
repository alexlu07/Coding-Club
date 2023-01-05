from random import gammavariate
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import scipy
import time
from tqdm import tqdm

from buffer import Buffer
from model import Model

class Trainer:
    def __init__(self, env, timesteps_per_batch=2048, minibatch_size=64, lr=3e-4, vf_coef=0.5,
                 n_steps = 10, gamma=0.99, lam=0.95, clip_ratio=0.2, rollout_device="cpu", 
                 train_device="cuda", temp=0.8):
        self.env = env
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.n

        # self.obs_dim = (obs_dim[2], obs_dim[0], obs_dim[1])
        self.obs_dim = (4,)
        self.act_dim = (act_dim,)

        self.rollout_device = rollout_device
        self.train_device = train_device

        self.epoch = 0

        self.timesteps_per_batch = timesteps_per_batch
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.vf_coef = vf_coef
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.temp = temp

        self.buffer = Buffer(self.obs_dim, self.act_dim, self.timesteps_per_batch, self.minibatch_size)

        self.model = Model(self.obs_dim, self.act_dim, env.observation_space.sample())
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def train_one_epoch(self):

        ep_lens = []
        ep_rets = []

        ep_len = 0
        ep_ret = 0

        start = time.time()

        self.model.to(self.rollout_device)

        obs = self.env.reset()[0]
        # obs = np.moveaxis(obs, 2, 0)

        with torch.no_grad():
            for step in tqdm(range(self.timesteps_per_batch), leave=False):
                obs_tensor = self.np_to_device(obs, self.rollout_device)
                act, val, logp = self.model.step(obs_tensor, self.temp)

                next_obs, rew, terminated, truncated, _ = self.env.step(act)
                done = terminated + truncated

                self.buffer.store(obs, act, rew, val, logp)

                ep_len += 1
                ep_ret += rew

                # next_obs = np.moveaxis(next_obs, 2, 0)
                obs = next_obs
                if done:
                    ep_lens.append(ep_len)
                    ep_rets.append(ep_ret)
                    
                    val = self.model.critic(self.np_to_device(obs, self.rollout_device)) if truncated else 0
                    self.buffer.finish_path(val)

                    obs = self.env.reset()[0]
                    # obs = np.moveaxis(obs, 2, 0)

                    ep_len = 0
                    ep_ret = 0

                    done = False

                elif step == self.timesteps_per_batch-1:
                    val = self.model.critic(self.np_to_device(obs, self.rollout_device))
                    self.buffer.finish_path(val.numpy())
    
        selfplay_time = time.time() - start
        start = time.time()

        self.model.to(self.train_device)

        batched_data = self.buffer.get(self.train_device)
        print(self.buffer.ret_buf[:50])
        print(self.buffer.val_buf[:50])


        for _ in range(self.n_steps):
            for data in tqdm(batched_data, leave=False):
                self.optimizer.zero_grad()
                loss = self.get_loss(data)[0]   
                loss.backward()
                self.optimizer.step()

        loss_old, loss_pi, loss_vf = [i.cpu().item() for i in self.get_loss(data)]

        training_time = time.time() - start

        self.epoch += 1

        return loss_pi, loss_vf, ep_lens, ep_rets, selfplay_time, training_time


    def get_loss(self, data):
        obs, act, adv, ret, logp_old = data['obs'], data['act'], data['adv'], data['ret'], data['logp']

        logp, val = self.model.chicken_nugget(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # print(ret, val)
    
        loss_vf = F.mse_loss(ret, val)

        loss = loss_pi + self.vf_coef * loss_vf
        return loss, loss_pi, loss_vf

    def save_state(self, name):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }, f"./weights/{name}.pt")

    def load_state(self, name):
        checkpoint = torch.load(f"./weights/{name}.pt")

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint["epoch"]

    def np_to_device(self, arr, device):
        return torch.as_tensor(arr, dtype=torch.float32).to(device)