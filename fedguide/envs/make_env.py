import importlib

def make_env(env_name: str):
    gym = None
    for mod in ("gymnasium", "gym"):
        if importlib.util.find_spec(mod):
            gym = importlib.import_module(mod)
            break
    if gym is None:
        raise ImportError("Please install gymnasium or gym")

    env = gym.make(env_name)

    def reset_env(e):
        out = e.reset()
        if isinstance(out, tuple):  # gymnasium
            obs, _info = out
        else:
            obs = out
        return obs

    def step_env(e, action):
        out = e.step(action)
        if len(out) == 5:  # gymnasium
            obs, reward, terminated, truncated, _info = out
            done = terminated or truncated
        else:              # old gym
            obs, reward, done, _info = out
        return obs, reward, done

    class EnvWrapper:
        def __init__(self, e):
            self.e = e
            self.observation_space = e.observation_space
            self.action_space = e.action_space

        def reset(self):
            return reset_env(self.e)

        def step(self, action):
            return step_env(self.e, action)

        def close(self):
            return self.e.close()

    return EnvWrapper(env)
