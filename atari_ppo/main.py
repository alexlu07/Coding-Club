import time
import gym
import numpy as np

from trainer import Trainer

env = gym.make("ALE/SpaceInvaders-v5")

trainer = Trainer(env)

save = True
load = False

# with open("./results/log.txt", "r+") as f:
#     if load:
#         epoch = int(f.readlines()[-1].split()[0][:-1])
#     else:
#         f.truncate(0)

# if save: trainer.save_state()``
# if load: trainer.load_state(epoch)

for i in range(100):
    start = time.time()
    loss, ep_lens, ep_rets, rollout_time, training_time = trainer.train_one_epoch()

    avg_rets = sum(ep_rets)/len(ep_rets)
    avg_lens = sum(ep_lens)/len(ep_lens)

    duration = time.time() - start

    log = f"{trainer.epoch}: loss={loss} episodes={{{avg_rets:.4f}, {avg_lens:.4f}}} time={{{duration:.4f}, {rollout_time:.4f}, {training_time:.4f}}}"
    print(log)

    # if save:
    #     trainer.save_state()
    #     with open("./results/log.txt", "a") as f:
    #         f.write(log + "\n")
trainer.model.to('cpu')

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
    if truncated:
        print("truncated")