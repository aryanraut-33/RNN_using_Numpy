import numpy as np
import numpy.random as rdm

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

'''
x_t -> input at a timestep t : [1,1]
y_t -> output at t : [1,1] or [1,C] where C = vocab size (possible no of outcomes), in case of textual prediction
h_t -> hidden state at t : [n_neurons,1]

W_i -> weights corresponding to inputs : [n_neurons,1]
W_o -> weights corresponding to the output at t : [1,n_neurons]
W_h -> weights corresponding to the hidden state at t : [n_neurons,n_neurons]
'''
class rnn_model():

    def __init__(self, X_t, n_neurons):

        self.T = max(X_t.shape)
        self.X_t = X_t
        
        self.Y_hat = np.zeros((self.T, 1))
        self.n_neurons = n_neurons

        self.Wh = 0.1*np.random.randn(n_neurons,n_neurons)
        self.Wi = 0.1*np.random.randn(n_neurons,1)
        self.Wo = 0.1*np.random.randn(1,n_neurons)
        self.bias = 0.1*np.random.randn(n_neurons,1)

        self.H = [np.zeros((n_neurons,1)) for t in range(self.T+1)]
    
    def forward(self, X_t, ht_1):
        out = np.dot(self.Wh,ht_1) + np.dot(self.Wi, X_t) + self.bias
        h_t = np.tanh(out)
        y_t = np.dot(self.Wo, h_t)
        return h_t, y_t, out
        





