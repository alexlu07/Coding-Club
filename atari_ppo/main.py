import time
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

from trainer import Trainer

writer = SummaryWriter(f"./tb_log/log")

env = gym.make("ALE/SpaceInvaders-v5")
# env = gym.make("CartPole-v1")

trainer = Trainer(env, temp=1.0, train_device="cpu")

for i in range(1000):
    start = time.time()
    loss_pi, loss_vf, ep_lens, ep_rets, rollout_time, training_time = trainer.train_one_epoch()

    avg_rets = sum(ep_rets)/len(ep_rets)
    avg_lens = sum(ep_lens)/len(ep_lens)

    duration = time.time() - start

    writer.add_scalar("rets", avg_rets, i)
    writer.add_scalar("lens", avg_lens, i)
    writer.add_scalar("loss_pi", loss_pi, i)
    writer.add_scalar("loss_vf", loss_vf, i)
    
trainer.model.to("cpu")
trainer.temp = 1.0

env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")

obs = env.reset()[0]
while True:
    obs = np.moveaxis(obs, 2, 0)
    obs = trainer.np_to_device(obs, 'cpu')

    action = trainer.model.step(obs)[0]
    obs, rewards, dones, truncated, info = env.step(action)

    env.render()
    if dones:
        print("done")
        break
    if truncated:
        print("truncated")
        break