import gym
import numpy as np


def make_reacher_split(client_id=0):
    """Different goal quadrants for different clients."""
    env = gym.make("Reacher-v4")
    theta = (client_id % 4) * np.pi / 2  # quadrant
    goal = np.array([np.cos(theta), np.sin(theta)]) * 0.1
    env.unwrapped.target = goal
    return env
