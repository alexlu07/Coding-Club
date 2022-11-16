import gym

from stable_baselines3 import PPO

env = gym.make("ALE/SpaceInvaders-v5")

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=30000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
