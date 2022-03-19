import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from cassie import CassieEnv

cassie = CassieEnv()
model = PPO("MlpPolicy", cassie, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = cassie.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = cassie.step(action)
    cassie.render()
