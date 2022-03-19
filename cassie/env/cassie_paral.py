import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from cassie import CassieEnv

def make_env(env_id):
    def _f():
        if env_id == 0:
            env = CassieEnv(visual=True)
        else:
            env = CassieEnv(visual=False)
        return env
    return _f

if __name__ == '__main__':
    envs =[make_env(seed) for seed in range(3)]
    envs = SubprocVecEnv(envs)

    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """
        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:                
            self.logger.record('reward/height', self.training_env.get_attr('qpos')[0][2])
            return True

    model = PPO("MlpPolicy", envs, verbose=1, n_steps=500,
        batch_size=128,tensorboard_log="./ppolog/")
    model.is_tb_set = False

    model.learn(total_timesteps=1e7,n_eval_episodes=10,callback=TensorboardCallback())
    model.save("ppo_cassie")

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_cassie")
    cassie = CassieEnv()
    obs = cassie.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = cassie.step(action)
        cassie.render()
