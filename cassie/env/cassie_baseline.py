import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from cassie import CassieEnv

cassie = CassieEnv()
model = PPO("MlpPolicy", cassie, verbose=1)
model.learn(total_timesteps=2000000)
model.save("ppo_cartpole")
read=input('model learned, press Enter to display')
obs = cassie.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = cassie.step(action)
read=input('model displayed, continue? y/n')

while True:
    if read == 'y':
        model.learn(total_timesteps=200000)
    else:
        break  
    read=input('model learned, press Enter to display')
    obs = cassie.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = cassie.step(action)
    read=input('model displayed, continue? y/n')



while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = cassie.step(action)
