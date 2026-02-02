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
#lr = 0.00001
optimizer = SGD_Optimizer(lr= 0.00001, momentum = 0.95, decay=0.01)

Monitor = np.zeros((epochs,1))

for n in range(epochs):
    
    rnn.forward()
    
    Y_hat = rnn.Y_hat
    dY = Y_hat - Y_t
    Loss = 0.5 * np.dot(dY.T, dY)/T ##Mean squared error
    
    Monitor[n] = Loss
    
    rnn.backward(dY)
    
    optimizer.pre_update_params()
    optimizer.update_params(rnn)
    optimizer.post_update_params()
    
    plt.plot(X_t, Y_t)
    plt.plot(X_t, Y_hat)
    plt.legend(['y', r'$\hat{y}$'])
    plt.title("epoch number: " + str(n))
    plt.show()
    

plt.plot(range(epochs), Monitor)
plt.xlabel('epoch #')
plt.ylabel('Loss')
plt.yscale('linear')
plt.show()






