import gym
import gymnasium as gymn
from fedguide.utils.gym_utils import reset_compat, make_d4rl_gymnasium

# 1) old/new gym
env_gym = gym.make("CartPole-v1")
obs, info = reset_compat(env_gym)
print("gym ok:", obs[:4])

# 2) gymnasium
env_gymn = gymn.make("CartPole-v1")
obs, info = reset_compat(env_gymn)
print("gymnasium ok:", obs[:4])

# 3) d4rl with gymnasium
env_d4rl_gn = make_d4rl_gymnasium("hopper-medium-v2")
dataset = env_d4rl_gn.get_dataset()
print("d4rl ok, dataset shape:", dataset["observations"].shape)

# step, should be gymnasium 5 outputs
obs, info = env_d4rl_gn.reset()
action = env_d4rl_gn.action_space.sample()
obs, reward, terminated, truncated, info = env_d4rl_gn.step(action)
print("step ok:", reward, terminated, truncated)