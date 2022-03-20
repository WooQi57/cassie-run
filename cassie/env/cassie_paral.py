import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from cassie import CassieRefEnv

def make_env(env_id):
    def _f():
        if env_id == 0:
            env = CassieRefEnv(visual=True)
        else:
            env = CassieRefEnv(visual=False)
        return env
    return _f

if __name__ == '__main__':
    envs =[make_env(seed) for seed in range(1)]
    envs = SubprocVecEnv(envs)

    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """
        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:                
            self.logger.record('reward/ref', self.training_env.get_attr('rew_ref')[0])
            self.logger.record('reward/spring', self.training_env.get_attr('rew_spring')[0])
            self.logger.record('reward/orientation', self.training_env.get_attr('rew_ori')[0])
            self.logger.record('reward/velocity', self.training_env.get_attr('rew_vel')[0])
            return True

    model = PPO("MlpPolicy", envs, verbose=1, n_steps=1024,
        batch_size=256,tensorboard_log="./ppolog/")
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
