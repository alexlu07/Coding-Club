import time
import gym
import numpy as np

from trainer import Trainer

# env = gym.make("ALE/SpaceInvaders-v5")
env = gym.make("CartPole-v1")

trainer = Trainer(env, lr=1e-3, temp=1.5)

for i in range(1000):
    start = time.time()
    loss, ep_lens, ep_rets, rollout_time, training_time = trainer.train_one_epoch()

    avg_rets = sum(ep_rets)/len(ep_rets)
    avg_lens = sum(ep_lens)/len(ep_lens)

    duration = time.time() - start

    log = f"{trainer.epoch}: loss={loss} episodes={{{avg_lens:.4f}, {avg_rets:.4f}}} time={{{duration:.4f}, {rollout_time:.4f}, {training_time:.4f}}}"
    print(log)
    with open("lr3temp15.log", "a") as f:
        f.write(log + "\n")

trainer.save_state("lr3temp15")

trainer.load_state("lr3temp15")
trainer.model.to("cpu")
trainer.temp = 1.0

env = gym.make("CartPole-v1", render_mode="human")

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