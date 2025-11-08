import gym
import gymnasium as gymn
import d4rl


def reset_compat(env):
    """ gym & gymnasium reset."""
    out = env.reset()
    if isinstance(out, tuple):  # gymnasium or gym>=0.26
        obs, info = out
    else:  
        obs, info = out, {}
    return obs, info


def step_compat(env, action):
    """5 return: (obs, reward, terminated, truncated, info)"""
    out = env.step(action)
    if len(out) == 5:  # gymnasium style
        obs, reward, terminated, truncated, info = out
        return obs, reward, terminated, truncated, info
    else:  # old gym style
        obs, reward, done, info = out
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

class GymToGymnasiumWrapper(gymn.Env):
    metadata = {"render_modes": []}

    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.observation_space = gym_env.observation_space
        self.action_space = gym_env.action_space

    def reset(self, **kwargs):
        obs, info = reset_compat(self.gym_env)
        return obs, info

    def step(self, action):
        return step_compat(self.gym_env, action)

    def render(self):
        return self.gym_env.render()

    def close(self):
        return self.gym_env.close()

    def get_dataset(self, *args, **kwargs):
        if hasattr(self.gym_env, "get_dataset"):
            return self.gym_env.get_dataset(*args, **kwargs)
        raise AttributeError("Underlying gym env has no get_dataset()")


def make_d4rl_gymnasium(env_id: str) -> gymn.Env:
    raw_env = gym.make(env_id) 
    return GymToGymnasiumWrapper(raw_env)