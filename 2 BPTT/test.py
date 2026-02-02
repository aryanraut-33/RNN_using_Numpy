# applying rnn to time-series data

import numpy as np
import matplotlib.pyplot as plt

#generating random time series data

X_t = np.arange(-10, 10, 0.1)
X_t = X_t.reshape(len(X_t), 1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t), 1)

n_neurons = 500

from rnn import *

rnn = RNN_model(X_t, n_neurons, Tanh())
rnn.forward()

T = rnn.T
H = rnn.H
Y_hat = rnn.Y_hat

dY = Y_hat - Y_t
Loss = 0.5 * np.dot(dY.T, dY)/T ##Mean squared error


for h in H:
    plt.plot(np.arange(20), h[0:20], 'k-', linewidth = 1, alpha = 0.05)


plt.plot(X_t, Y_t)
plt.plot(X_t, Y_hat)
plt.legend(['y', r'$\hat{y}$'])
plt.show()







