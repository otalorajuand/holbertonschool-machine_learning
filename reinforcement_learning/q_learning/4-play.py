#!/usr/bin/env python3
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def play(env, Q, max_steps=100):
    """has the trained agent play an episode

    Args:
    - env is the FrozenLakeEnv instance
    - Q is a numpy.ndarray containing the Q-table
    - max_steps is the maximum number of steps in the episode

    Returns: the total rewards for the episode
    """
    state = env.reset()[0]

    current_row = 0
    current_col = 0

    map_desc = env.spec.kwargs['desc']
    rows = len(map_desc)
    cols = len(map_desc[0])

    for row in range(rows):
        for col in range(cols):
            if row == current_row and col == current_col:
                print(f"`{map_desc[row][col]}`", end='')
            else:
                print(map_desc[row][col], end='')
        print()  # Start a new line for the next row

    for step in range(max_steps):

        action = np.argmax(Q[state, :])
        new_state, reward, done, info, _ = env.step(action)

        if action == 0:
            current_col -= 1
            print('    (Left)')
        elif action == 1:
            current_row += 1
            print('    (Down)')
        elif action == 2:
            current_col += 1
            print('    (Right)')
        else:
            current_row -= 1
            print('    (Up)')

        for row in range(rows):
            for col in range(cols):
                if row == current_row and col == current_col:
                    print(f"`{map_desc[row][col]}`", end='')
                else:
                    print(map_desc[row][col], end='')
            print()  # Start a new line for the next row

        state = new_state

        if done:
            break

    return reward
