import time
import gym

from trainer import Trainer

env = gym.make("ALE/Breakout-v5")

trainer = Trainer(env)

# with open("./results/log.txt", "r+") as f:
#     if load:
#         epoch = int(f.readlines()[-1].split()[0][:-1])
#     else:
#         f.truncate(0)

# if save: trainer.save_state()``
# if load: trainer.load_state(epoch)

while True:
    start = time.time()
    loss, ep_lens, ep_rets, selfplay_time, training_time = trainer.train_one_epoch()

    avg_rets = sum(ep_rets)/len(ep_rets)
    avg_lens = sum(ep_lens)/len(ep_lens)

    duration = time.time() - start

    log = f"{trainer.epoch}: loss={loss} episodes={{{avg_rets:.4f}, {avg_lens:.4f}}} time={{{duration:.4f}, {selfplay_time:.4f}, {training_time:.4f}}}"
    print(log)

    # if save:
    #     trainer.save_state()
    #     with open("./results/log.txt", "a") as f:
    #         f.write(log + "\n")
