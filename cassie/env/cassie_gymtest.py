from cassie import CassieEnv
from stable_baselines3.common.env_checker import check_env

cassie = CassieEnv()
check_env(cassie)

print('done')
