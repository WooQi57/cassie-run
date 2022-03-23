import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from cassie import CassieRefEnv

def make_env(env_id):
    def _f():
        if env_id == 0:
            env = CassieRefEnv(visual=False, dynamics_randomization=False)
        else:
            env = CassieRefEnv(visual=False, dynamics_randomization=False)
        return env
    return _f

if __name__ == '__main__':
    envs =[make_env(seed) for seed in range(16)]
    envs = SubprocVecEnv(envs)

    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """
        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:                
            self.logger.record('reward/ref', self.training_env.get_attr('rew_ref_buf')[0])
            self.logger.record('reward/spring', self.training_env.get_attr('rew_spring_buf')[0])
            self.logger.record('reward/orientation', self.training_env.get_attr('rew_ori_buf')[0])
            self.logger.record('reward/velocity', self.training_env.get_attr('rew_vel_buf')[0])
            self.logger.record('reward/termination', self.training_env.get_attr('rew_termin_buf')[0])
            self.logger.record('reward/steps', self.training_env.get_attr('time_buf')[0])
            self.logger.record('reward/totalreward', self.training_env.get_attr('reward_buf')[0])

            if self.n_calls % 51200 == 0:
                print("Saving new best model")
                self.model.save(f"./model_saved/ppo_cassie_{self.n_calls}")

            return True

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[512, 512], vf=[512, 512])])
    model = PPO("MlpPolicy", envs, verbose=1, n_steps=512, policy_kwargs=policy_kwargs,
        batch_size=256,tensorboard_log="./ppolog/")
    model.is_tb_set = False

    model.learn(total_timesteps=1e7,n_eval_episodes=10,callback=TensorboardCallback())
    model.save("ppo_cassie")

