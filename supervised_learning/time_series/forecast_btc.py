import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from window import WindowGenerator

# Create a WindowGenerator instance for a wide window
wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['Close'])

# Define an LSTM model using TensorFlow's Sequential API
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

# Set the maximum number of epochs for training
MAX_EPOCHS = 20


def compile_and_fit(model, window, patience=2):
    """
    Compile and fit the model on the given window data.

    Args:
        model: TensorFlow model.
        window: WindowGenerator instance.
        patience: Number of epochs with no improvement after which training
                  will be stopped.

    Returns:
        history: Training history.
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


# Compile and fit the LSTM model using the wide window data
history = compile_and_fit(lstm_model, wide_window)

# Evaluate the model's performance on validation data
val_performance = {}
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)

# Evaluate the model's performance on test data
performance = {}
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
