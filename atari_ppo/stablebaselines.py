import gym

from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1,  tensorboard_log="tb_log/")
print(model.policy)
model.learn(total_timesteps=20000, tb_log_name="sb_log")

buffer = model.rollout_buffer
print(buffer.returns[:40])
print(model.policy(obs_as_tensor(buffer.observations, "cuda"))[1][:40])
print(buffer.observations[0])

# del model # remove to demonstrate saving and loading

# env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
# model = PPO.load("ppo_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
