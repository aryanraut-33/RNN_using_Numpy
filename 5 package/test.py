# -*- coding: utf-8 -*-

# applying rnn to time-series data

import numpy as np
import matplotlib.pyplot as plt

###########################################################

'''
Training the model:
    - This is the Training function. 
    - Its job is to "teach" the network the pattern in your data.
    - It performs Forward Propagation (making a guess), calculates errors, performs Backpropagation (learning from errors), and Updates Weights.
    - Inputs: It needs both the input data X_t and the correct answers Y_t.
    - Returns: The trained rnn object (which now holds the learned weights).
'''

#generating random time series data

X_t = np.arange(-100, 100, 0.5) 
X_t = X_t.reshape(len(X_t), 1)
Y_t = np.sin(0.05 * X_t) + 0.1*np.random.randn(len(X_t), 1) + 0.2*np.exp(0.01*X_t) # Adjusted frequencies for larger scale

from rnn import *


rnn = Run_RNN(X_t, Y_t, Tanh(), plot_each = 0.50, epochs = 500, decay=0)

###########################################################

'''
Applying the model on unseen data: 
    - This is the Prediction (or Inference) function. 
    - Its job is to use the already learned weights to make predictions on new data.
    - It takes a trained rnn model as input.
    - It runs only the Forward Propagation step using rnn.RNN_Cell.
    - It does not calculate errors, does not run backpropagation, and does not change any weights.
    - Inputs: It only needs input data X_t (and the trained rnn).
    - Returns: The predictions Y_hat.
'''

X_new = np.arange(-20, 50, 0.5) 
X_new = X_new.reshape(len(X_new), 1)

Y_hat = Apply_RNN(X_new, rnn)

plt.plot(X_t, Y_t)
plt.plot(X_new, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()

###########################################################

'''
Autoregressive Training:
    - Here we teach the network to predict the next value in a sequence based on past values.
    - Key Difference: It uses the same Y_t data for both input and target, but shifted by a time lag dt.
    - Input: Y_t (up to T-dt).
    - Target: Y_t (from dt onwards).
    - Goal: Learn Y_{t} = f(Y_{t-dt}).
'''
dt = 20

rnn = Run_RNN(Y_t, Y_t, Tanh(), epochs=500, n_neurons=100, decay=0.01, dt = dt)

Y_hat = Apply_RNN(Y_t, rnn)

X_t = np.arange(len(Y_t))

plt.plot(X_t, Y_t)
plt.plot(X_t, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()