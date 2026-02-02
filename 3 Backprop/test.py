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

T = rnn.T
epochs = 200
lr = 0.00001

for n in range(epochs+1):
    
    rnn.forward()
    
    Y_hat = rnn.Y_hat
    dY = Y_hat - Y_t
    Loss = 0.5 * np.dot(dY.T, dY)/T ##Mean squared error
    
    print(float(Loss.item()))
    
    rnn.backward(dY)
    
    rnn.Wi -= lr * rnn.d_Wi
    rnn.Wo -= lr * rnn.d_Wo
    rnn.Wh -= lr * rnn.d_Wh
    rnn.bias -= lr * rnn.d_bias
    
    plt.plot(X_t, Y_t)
    plt.plot(X_t, Y_hat)
    plt.legend(['y', r'$\hat{y}$'])
    plt.title("epoch number: " + str(n))
    plt.show()








