# Q-learning

> This project delves into Q-learning in reinforcement learning. It covers fundamental concepts and tasks such as initializing a FrozenLake environment, initializing a Q-table, implementing epsilon-greedy exploration, performing Q-learning, and testing the learned Q-table by having the trained agent play the game. Each task involves code implementation, aiming to understand and apply Q-learning principles via Python functions, using OpenAI's Gym library and FrozenLake environment. The project aims to reinforce knowledge on Q-learning basics, environment manipulation, Q-table updating, and policy decisions to train an agent efficiently in a simple game environment.

At the end of this project I was able to solve these conceptual questions:

* What is a Markov Decision Process?
* What is an environment?
* What is an agent?
* What is a state?
* What is a policy function?
* What is a value function? a state-value function? an action-value function?
* What is a discount factor?
* What is the Bellman equation?
* What is epsilon greedy?
* What is Q-learning?

## Tasks :heavy_check_mark:

| Filename | Task |
| ------ | ------------------------------------------------- | 
| [0-load_env.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/reinforcement_learning/q_learning/0-load_env.py)| The code defines a Python function to load the FrozenLake environment from OpenAI's Gym, allowing custom maps, map selection, and ice slipperiness settings. | 
| [1-q_init.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/reinforcement_learning/q_learning/1-q_init.py)| The code initializes a Q-table as a NumPy array of zeros for a FrozenLake environment. | 
| [2-epsilon_greedy.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/reinforcement_learning/q_learning/2-epsilon_greedy.py)| The code implements epsilon-greedy strategy for the next action in Q-learning. | 
| [3-q_learning.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/reinforcement_learning/q_learning/3-q_learning.py)| The code performs Q-learning iterations using the epsilon-greedy strategy in the FrozenLake environment. | 
| [4-play.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/reinforcement_learning/q_learning/4-play.py)| The code allows a trained agent to navigate and play an episode in the FrozenLake environment. | 

### Try It On Your Machine :computer:
```bash
git clone https://github.com/otalorajuand/holbertonschool-machine_learning.git
cd reinforcement_learning/q_learning
```
