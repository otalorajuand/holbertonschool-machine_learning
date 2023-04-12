#!/usr/bin/env python3
"""This module includes the Class NeuralNetwork"""
import numpy as np
import matplotlib.pyplot as plt



class NeuralNetwork:
    """this class defines a neural network with one hidden layer
    performing binary classification"""

    def __init__(self, nx, nodes):

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.normal(0, 1, size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """This function Calculates the forward propagation
        of the neural network"""
        Z1 = np.dot(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using neural network"""
        loss = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -(np.sum(loss) / loss.shape[1])
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A = np.where(self.forward_prop(X)[1] < 0.5, 0, 1)
        return A, self.cost(Y, self.forward_prop(X)[1])

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        DZ2 = A2 - Y
        m = X.shape[1]
        DW2 = (1 / m) * np.dot(DZ2, A1.T)
        DB2 = (1 / m) * np.sum(DZ2, axis=1, keepdims=True)

        DZ1 = np.dot(self.__W2.T, DZ2) * (A1 * (1 - A1))
        DW1 = (1 / m) * np.dot(DZ1, X.T)
        DB1 = (1 / m) * np.sum(DZ1, axis=1, keepdims=True)

        self.__W2 = self.__W2 - alpha * DW2
        self.__b2 = self.__b2 - alpha * DB2

        self.__W1 = self.__W1 - alpha * DW1
        self.__b1 = self.__b1 - alpha * DB1


    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the neural network"""
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')

        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        if verbose or graph:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step <=0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        costs = []
        for i in range(iterations+1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)

            if verbose and i%step == 0 or i in [0, iterations]:
                print(f'Cost after {i} iterations: {self.cost(Y, self.A2)}')
    
            if graph and i%step == 0 or i in [0, iterations]: 
                costs.append(self.cost(Y, self.A2))

        if graph:
            x = np.arange(0, iterations+1, step)
            y = np.array(costs)
            plt.plot(x, y)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
