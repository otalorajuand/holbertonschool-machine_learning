#!/usr/bin/env python3
import numpy as np


def q_init(env):
    """initializes the Q-table

    Args:
    - env is the FrozenLakeEnv instance

    Returns: the Q-table as a numpy.ndarray of zeros
    """

    cols = env.action_space.n
    rows = env.observation_space.n

    return np.zeros((rows, cols))
