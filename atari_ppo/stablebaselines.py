import gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1,  tensorboard_log="tb_log/")
print(model.policy)
model.learn(total_timesteps=1000000, tb_log_name="sb_log")
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
