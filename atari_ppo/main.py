import time
import gymnasium as gym
from gymnasium.wrappers import *
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

from trainer import Trainer

writer = SummaryWriter(f"./tb_log/log")

env = gym.make("ALE/SpaceInvaders-v5", frameskip=1)
env = TimeLimit(FrameStack(AtariPreprocessing(env, frame_skip=2, grayscale_newaxis=False, terminal_on_life_loss=True), 4), max_episode_steps=2000)
# env = gym.make("CartPole-v1")

trainer = Trainer(env, temp=2.0, rollout_device="cuda", train_device="cuda")

i = 0
while True:
    start = time.time()
    loss_pi, loss_vf, ep_lens, ep_rets, rollout_time, training_time = trainer.train_one_epoch()

    avg_rets = sum(ep_rets)/len(ep_rets)
    avg_lens = sum(ep_lens)/len(ep_lens)

    duration = time.time() - start

    writer.add_scalar("rets", avg_rets, i)
    writer.add_scalar("lens", avg_lens, i)
    writer.add_scalar("loss_pi", loss_pi, i)
    writer.add_scalar("loss_vf", loss_vf, i)
    
    if i % 100 == 0 and i > 0:
        trainer.model.to("cpu")
        trainer.save_state(f"ent{i}")

    i += 1