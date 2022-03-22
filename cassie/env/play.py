import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from cassie import CassieRefEnv



if __name__ == '__main__':
    t = time.monotonic()
    model = PPO.load("model_saved/ppo_cassie_1228800")
    cassie = CassieRefEnv(dynamics_randomization=False)
    obs = cassie.reset()
    # print(obs)
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = cassie.step(action)
        while time.monotonic() - t < 60*0.0005:
            time.sleep(0.0001)
        t = time.monotonic()
        cassie.render()
        if dones:
            obs = cassie.reset()
            # print(obs)
