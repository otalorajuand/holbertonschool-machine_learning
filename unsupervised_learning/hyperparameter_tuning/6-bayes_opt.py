#!/usr/bin/env python3
#Keras
import tensorflow.keras as K

#GPyOpt - Cases are important, for some reason
import GPyOpt
from GPyOpt.methods import BayesianOptimization

#nump
import numpy as np



def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function that builds a neural network with the Keras library
    Args:
      nx is the number of input features to the network
      layers is a list containing the number of nodes in each layer of the
      network
      activations is a list containing the activation functions used for
      each layer of the network
      lambtha is the L2 regularization parameter
      keep_prob is the probability that a node will be kept for dropout
    Returns: the keras model
    """
    inputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(float(lambtha))

    output = K.layers.Dense(layers[0],
                            activation=activations[0],
                            kernel_regularizer=regularizer)(inputs)

    hidden_layers = range(len(layers))[1:]

    for i in hidden_layers:
        dropout = K.layers.Dropout(1 - float(keep_prob))(output)
        output = K.layers.Dense(layers[i], activation=activations[i],
                                kernel_regularizer=regularizer)(dropout)

    model = K.Model(inputs, output)

    return model

def optimize_model(network, alpha, beta1, beta2):
    """
    Function that sets up Adam optimization for a keras model with categorical
    crossentropy loss and accuracy metrics
    Args:
    network is the model to optimize
    alpha is the learning rate
    beta1 is the first Adam optimization parameter
    beta2 is the second Adam optimization parameter
    Returns: None
    """
    Adam = K.optimizers.Adam(learning_rate=float(alpha),
                             beta_1=float(beta1),
                             beta_2=beta2)

    network.compile(optimizer=Adam,
                    loss="categorical_crossentropy",
                    metrics=['accuracy'])
    

def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, filepath=None,
                verbose=False, shuffle=False):
    """
    Function That trains a model using mini-batch gradient descent
    Args:
    network is the model to train
    data is a numpy.ndarray of shape (m, nx) containing the input data
    labels is a one-hot numpy.ndarray of shape (m, classes) containing
    the labels of data
    batch_size is the size of the batch used for mini-batch gradient descent
    epochs is the number of passes through data for mini-batch gradient descent
    validation_data is the data to validate the model with, if not None
    """
    def learning_rate_decay(epoch):
        """Function tha uses the learning rate"""
        alpha_0 = alpha / (1 + (decay_rate * epoch))
        return alpha_0

    callbacks = []

    if validation_data:
        if early_stopping:
            early_stop = K.callbacks.EarlyStopping(patience=patience)
            callbacks.append(early_stop)


        if learning_rate_decay:
            decay = K.callbacks.LearningRateScheduler(learning_rate_decay,
                                                      verbose=verbose)
            callbacks.append(decay)

    
    if filepath:
        save = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        callbacks.append(save)

    train = network.fit(x=data,
                        y=labels,
                        batch_size=int(batch_size),
                        epochs=epochs,
                        validation_data=validation_data,
                        callbacks=callbacks,
                        verbose=False,
                        shuffle=shuffle)

    return train

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot.T


def object_function(x):
        """
        Function that set hyperparameters of a keras network:
            Args: X is a vector conating the parameter to optimized and trained
                lambtha is the L2 regularization parameter
                keep_prob is the probability that a node will be kept for dropout
                alpha is the learning rate in Adam optimizer
                beta1 is the first Adam optimization parameter
                batch_size is the size of the batch used for mini-batch  gradient descent
            Returns the loss of the model
        """
        # x is 5 dimentional vector with the parameter we want to optimize
        lambtha = x[:, 0]
        keep_prob = x[:, 1]
        alpha = x[:, 2]
        beta1 = x[:, 3]
        batch_size = x[:, 4]

        # Exporting the data, for handwirte numbers database
        datasets = np.load('../../supervised_learning/data/MNIST.npz')
        X_train = datasets['X_train']
        X_train = X_train.reshape(X_train.shape[0], -1)
        Y_train = datasets['Y_train']
        Y_train_oh = one_hot(Y_train, 10)
        X_valid = datasets['X_valid']
        X_valid = X_valid.reshape(X_valid.shape[0], -1)
        Y_valid = datasets['Y_valid']
        Y_valid_oh = one_hot(Y_valid, 10)

        # Building the model using Keras library
        network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)

        # Optimizing the model using adam optimizer
        beta2 = 0.999
        optimize_model(network, alpha, beta1, beta2)

        # Training the model using early stopping and saving the best modle in bayes_opt.txt'
        epochs = 100
        history = train_model(network, X_train, Y_train_oh, batch_size, epochs,
                              validation_data=(X_valid, Y_valid_oh), early_stopping=True,
                              patience=3, learning_rate_decay=True)

        return (history.history['val_loss'][-1])


# Setting the bounds of network parameter for the bayeyias optimizatio
bounds = [{'name': 'lambtha', 'type': 'continuous','domain': (0.0001, 0.0005)},
            {'name': 'keep_prob', 'type': 'continuous','domain': (0.80, 0.95)},
            {'name': 'alpha', 'type': 'continuous','domain': (0.001, 0.005)},
            {'name': 'beta1', 'type': 'continuous', 'domain': (0.9, 0.99)},
            {'name': 'batch_size', 'type': 'discrete', 'domain': (50, 70)}]


# Creating the GPyOpt method using Bayesian Optimizatio
my_Bayes_opt = GPyOpt.methods.BayesianOptimization(object_function, domain=bounds, verbosity=True)


#Stop conditions
max_time  = None 
max_iter  = 30
tolerance = 1e-8


#Running the method
my_Bayes_opt.run_optimization(max_iter = max_iter,
                              max_time = max_time,
                              eps = tolerance)

print("Value of (x,y) that minimises the objective:"+str(my_Bayes_opt.x_opt))    
print("Minimum value of the objective: "+str(my_Bayes_opt.fx_opt))  