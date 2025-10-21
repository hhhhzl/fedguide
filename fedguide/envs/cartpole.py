import gymnasium as gym


def make_cartpole_sparse():
    env = gym.make("CartPole-v1")
    env.spec.reward_threshold = 475

    def step(a):
        s, r, done, _, info = env.step(a)
        r = 1.0 if done and env.steps_beyond_terminated is None else 0.0
        return s, r, done, False, info

    env.step = step
    return env


def make_mountaincar_sparse():
    env = gym.make("MountainCar-v0")

    def step(a):
        s, r, done, _, info = env.step(a)
        r = 1.0 if done else 0.0
        return s, r, done, False, info

    env.step = step
    return env
