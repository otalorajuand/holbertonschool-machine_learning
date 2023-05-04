# Optimization

> Optimization is the process which allow us to update the parameters of the model in order to reduce an error metric. The most used methodology to update the parameters of a neural network is called gradient descente. In this project we explore the improvements of this classic algorithm to make it faster and more efficient.

At the end of this project I was able to solve these conceptual questions:

* What is a hyperparameter?
* How and why do you normalize your input data?
* What is a saddle point?
* What is stochastic gradient descent?
* What is mini-batch gradient descent?
* What is a moving average? How do you implement it?
* What is gradient descent with momentum? How do you implement it?
* What is RMSProp? How do you implement it?
* What is Adam optimization? How do you implement it?
* What is learning rate decay? How do you implement it?
* What is batch normalization? How do you implement it?

## Tasks :heavy_check_mark:

0. Write the function that calculates the normalization (standardization) constants of a matrix.
1. Write the function that normalizes (standardizes) a matrix.
2. Write the function that shuffles the data points in two matrices the same way.
3. Write the function that trains a loaded neural network model using mini-batch gradient descent.
4. Write the function that calculates the weighted moving average of a data set.
5. Write the function that updates a variable using the gradient descent with momentum optimization algorithm.
6. Write the function that creates the training operation for a neural network in tensorflow using the gradient descent with momentum optimization algorithm.
7. Write the function that updates a variable using the RMSProp optimization algorithm.
8. Write the function that creates the training operation for a neural network in tensorflow using the RMSProp optimization algorithm.
9. Write the function that updates a variable in place using the Adam optimization algorithm.
10. Write the function that creates the training operation for a neural network in tensorflow using the Adam optimization algorithm.
11. Write the function that updates the learning rate using inverse time decay in numpy.
12. Write the function that creates a learning rate decay operation in tensorflow using inverse time decay.
13. Write the function that normalizes an unactivated output of a neural network using batch normalization.
14. Write the function that creates a batch normalization layer for a neural network in tensorflow.
15. Write the function that builds, trains, and saves a neural network model in tensorflow using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization.
16. Write a blog post explaining the mechanics, pros, and cons of the following optimization techniques:

- Feature Scaling
- Batch normalization
- Mini-batch gradient descent
- Gradient descent with momentum
- RMSProp optimization
- Adam optimization
- Learning rate decay

## Results :chart_with_upwards_trend:

| Filename | Task |
| ------ | ------------------------------------------------- | 
| [0-norm_constants.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/0-norm_constants.py)| Write the function that calculates the normalization (standardization) constants of a matrix.| 
| [1-normalize.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/1-normalize.py)| Write the function that normalizes (standardizes) a matrix.|
| [2-shuffle_data.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/2-shuffle_data.py)| Write the function that shuffles the data points in two matrices the same way.|
| [3-mini_batch.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/3-mini_batch.py)| Write the function that trains a loaded neural network model using mini-batch gradient descent.|
| [4-moving_average.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/4-moving_average.py)| Write the function that calculates the weighted moving average of a data set.|
| [5-momentum.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/5-momentum.py)| Write the function that updates a variable using the gradient descent with momentum optimization algorithm.|
| [6-momentum.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/6-momentum.py)| Write the function that creates the training operation for a neural network in tensorflow using the gradient descent with momentum optimization algorithm.|
| [7-RMSProp.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/7-RMSProp.py)| Write the function that updates a variable using the RMSProp optimization algorithm.|
| [8-RMSProp.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/8-RMSProp.py)| Write the function that creates the training operation for a neural network in tensorflow using the RMSProp optimization algorithm.|
| [9-Adam.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/9-Adam.py)| Write the function that updates a variable in place using the Adam optimization algorithm.|
| [10-Adam.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/10-Adam.py)| Write the function that creates the training operation for a neural network in tensorflow using the Adam optimization algorithm.|
| [11-learning_rate_decay.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/11-learning_rate_decay.py)| Write the function that updates the learning rate using inverse time decay in numpy.|
| [12-learning_rate_decay.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/12-learning_rate_decay.py)| Write the function that creates a learning rate decay operation in tensorflow using inverse time decay.|
| [13-batch_norm.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/13-batch_norm.py)| Write the function that normalizes an unactivated output of a neural network using batch normalization.|
| [14-batch_norm.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/14-batch_norm.py)| Write the function that creates a batch normalization layer for a neural network in tensorflow.|
| [15-model.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/optimization/15-model.py)| Write the function that builds, trains, and saves a neural network model in tensorflow using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization.|


### Try It On Your Machine :computer:
```bash
git clone https://github.com/otalorajuand/holbertonschool-machine_learning.git
cd optimization
```
