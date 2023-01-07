import time
import gymnasium as gym
from gymnasium.wrappers import *
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

from trainer import Trainer

env = gym.make("ALE/SpaceInvaders-v5", frameskip=1, render_mode="human")
env = TimeLimit(FrameStack(AtariPreprocessing(env, frame_skip=2, grayscale_newaxis=False, terminal_on_life_loss=True), 4), max_episode_steps=2000)

trainer = Trainer(env, temp=1.0, train_device="cpu")
trainer.load_state(input("name: "))

trainer.model.to("cpu")

obs = env.reset()[0]
while True:
    # obs = np.moveaxis(obs, 2, 0)
    obs = np.array(obs)
    obs = trainer.np_to_device(obs, 'cpu')

    action = trainer.model.step(obs, temp=0.1)[0]
    obs, rewards, dones, truncated, info = env.step(action)
    print(info)

    env.render()
    if dones:
        print("done")
        break
    if truncated:
        print("truncated")
        break