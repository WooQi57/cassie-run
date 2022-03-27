import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from cassie import CassieRefBuf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    t = time.monotonic()
    model = PPO.load("model_saved/ppo_cassie_"+str(512 * 11)+"00")
    # model = PPO.load("ppo_cassie")
    cassie = CassieRefBuf(dynamics_randomization=False)
    obs = cassie.reset()
    vel = []
    print(obs.shape)

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = cassie.step(action)
        while time.monotonic() - t < 60*0.0005:
            time.sleep(0.0001)
        t = time.monotonic()
        cassie.render()
        vel.append(cassie.qvel[0])
        # print(cassie.foot_pos[2::3],cassie.phase,cassie.custom_footheight())
        # print(cassie.qvel[3:6])
        if dones:
            plt.plot([cassie.simrate/2000*x for x in range(len(vel))],vel)
            plt.axhline(y=cassie.speed,c='r')
            plt.legend(['robot','command'])
            plt.ylabel('velo_x (m/s)')
            plt.xlabel('time (s)')
            plt.show()
            vel = []
            obs = cassie.reset()
