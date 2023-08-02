# Hyperparameter Tuning

> In this project we explore bayesian optimization for hyperparameter selection in neural networks. We also check Gaussian process which are needed to implement the first algorithm.

At the end of this project I was able to solve these conceptual questions:

* What is Hyperparameter Tuning?
* What is random search? grid search?
* What is a Gaussian Process?
* What is a mean function?
* What is a Kernel function?
* What is Gaussian Process Regression/Kriging?
* What is Bayesian Optimization?
* What is an Acquisition function?
* What is Expected Improvement?
* What is Knowledge Gradient?
* What is Entropy Search/Predictive Entropy Search?
* What is GPy?
* What is GPyOpt?

## Tasks :heavy_check_mark:

| Filename | Task |
| ------ | ------------------------------------------------- | 
| [0-gp.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/unsupervised_learning/hyperparameter_tuning/0-gp.py)| Create the class GaussianProcess that represents a noiseless 1D Gaussian process. | 
| [1-gp.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/unsupervised_learning/hyperparameter_tuning/1-gp.py)| Based on 0-gp.py, update the class GaussianProcess to add the prediction method. | 
| [2-gp.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/unsupervised_learning/hyperparameter_tuning/2-gp.py)| Based on 1-gp.py, update the class GaussianProcess to add the update method. | 
| [3-bayes_opt.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/unsupervised_learning/hyperparameter_tuning/3-bayes_opt.py)| Create the class BayesianOptimization that performs Bayesian optimization on a noiseless 1D Gaussian process. | 
| [4-bayes_opt.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/unsupervised_learning/hyperparameter_tuning/4-bayes_opt.py)| Based on 3-bayes_opt.py, update the class BayesianOptimization to add the method acquisition. | 
| [5-bayes_opt.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/unsupervised_learning/hyperparameter_tuning/5-bayes_opt.py)| Based on 4-bayes_opt.py, update the class BayesianOptimization to add the method optimize. | 


### Try It On Your Machine :computer:
```bash
git clone https://github.com/otalorajuand/holbertonschool-machine_learning.git
cd unsupervised_learning/hyperparameter_tuning
```