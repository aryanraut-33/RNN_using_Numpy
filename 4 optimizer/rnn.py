import numpy as np

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
class RNN_model():

    def __init__(self, X_t, n_neurons, Activation):

        self.T = max(X_t.shape)
        self.X_t = X_t
        
        self.Y_hat = np.zeros((self.T, 1))
        self.n_neurons = n_neurons

        # set up the matrices for the learnable parameters
        self.Wh = 0.1*np.random.randn(n_neurons,n_neurons)
        self.Wi = 0.1*np.random.randn(n_neurons,1)
        self.Wo = 0.1*np.random.randn(1,n_neurons)
        self.bias = 0.1*np.random.randn(n_neurons,1)

        self.H = [np.zeros((n_neurons,1)) for t in range(self.T+1)]
        self.Activation = Activation
    
    def forward(self):

        # Initialize gradients to zero to prevent accumulation from previous training iterations.
        # This ensures we start with a clean state for the current pass before accumulating gradients during BPTT.

        self.d_Wh = np.zeros((self.n_neurons,self.n_neurons))
        self.d_Wi = np.zeros((self.n_neurons,1))
        self.d_Wo = np.zeros((1,self.n_neurons))
        self.d_bias = np.zeros((self.n_neurons,1))

        X_t = self.X_t
        H = self.H
        Y_hat = self.Y_hat

        h_t = H[0] #inintal state vector

        Activation = self.Activation 

        Act_list = [Activation for t in range(self.T)]    

        [Act_list, H, Y_hat] = self.RNN_Cell(X_t, h_t, Act_list, H, Y_hat)

        self.H = H
        self.Act_list = Act_list
        self.Y_hat = Y_hat
        
    def backward(self, d_values): #d_values :---> directed from the loss function

        T = self.T
        H = self.H
        Act_list = self.Act_list
        X_t = self.X_t

        Act_list = self.Act_list
        
        Wh = self.Wh
        Wo = self.Wo

        d_Wh = self.d_Wh
        d_Wi = self.d_Wi
        d_Wo = self.d_Wo
        d_bias = self.d_bias

        d_h_t = np.dot(Wo.T, d_values[-1].reshape(1,1))

        for t in reversed(range(T)): #forward prop just reversed!

            d_y = d_values[t].reshape(1,1)
            x_t = X_t[t].reshape(1,1)
            
            Act_list[t].backward(d_h_t)
            dtanh = Act_list[t].d_inputs
            
            d_Wi = d_Wi + np.dot(dtanh, x_t)
            d_Wo = d_Wo + np.dot(H[t+1], d_y).T
            d_Wh = d_Wh + np.dot(H[t], dtanh.T)
            d_bias = d_bias + dtanh

            #change of ht comes from 2 sources : 1. directly from loss function & 2. from the previous cell/timestamp
            if t > 0:
                d_h_t = np.dot(Wh.T, dtanh) + np.dot(Wo.T, d_values[t-1].reshape(1,1))
            else:
                d_h_t = np.dot(Wh.T, dtanh)


        self.d_Wh = d_Wh
        self.d_Wi = d_Wi
        self.d_Wo = d_Wo
        self.d_bias = d_bias
        
        self.H = H

        

    def RNN_Cell(self, X_t, h_t, Act_list, H, Y_hat):

        for t , xt in enumerate(X_t):
            xt = xt.reshape(1,1)
            out = np.dot(self.Wh,h_t) + np.dot(self.Wi, xt) + self.bias
            Act_list[t].forward(out)
            h_t = Act_list[t].output
            y_hat_t = np.dot(self.Wo, h_t) #prediction at time t
            
            #save the records
            H[t+1] = h_t
            Y_hat[t] = y_hat_t

        return Act_list, H, Y_hat


### Defining activation [tanh] and its derivative

class Tanh():

    def forward(self, inputs):
        self.output = np.tanh(inputs)
        self.inputs = inputs
    
    def backward(self, d_values):
        derivative = 1 - self.output**2
        self.d_inputs = d_values * derivative
        

### Defining SGD optimizer

class SGD_Optimizer:
    
    def __init__(self, lr = 0.00001, momentum = 0, decay = 0):
        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.current_lr = lr
        self.iterations = 0
    
    def pre_update_params(self):  ##incase there is a decay value ≠ 0
        if self.decay:
            self.current_lr = self.lr * (1/(1 + (self.decay * self.iterations)))
            
    def update_params(self, layer):
        
        if self.momentum: ## incase there is a momentum value ≠ 0
             
            if not hasattr(layer, 'Wi_momentums'):
                layer.Wi_momentums = np.zeros_like(layer.Wi)
                layer.Wo_momentums = np.zeros_like(layer.Wo)
                layer.Wh_momentums = np.zeros_like(layer.Wh)
                layer.bias_momentums = np.zeros_like(layer.bias)
                
            Wi_updates = self.momentum * layer.Wi_momentums - (self.current_lr * layer.d_Wi)
            layer.Wi_momentums = Wi_updates
            
            Wo_updates = self.momentum * layer.Wo_momentums - (self.current_lr * layer.d_Wo)
            layer.Wo_momentums = Wo_updates
            
            Wh_updates = self.momentum * layer.Wh_momentums - (self.current_lr * layer.d_Wh)
            layer.Wh_momentums = Wh_updates
            
            bias_updates = self.momentum * layer.bias_momentums - (self.current_lr * layer.d_bias)
            layer.bias_momentums = bias_updates
        
        else:
            Wi_updates = -self.current_lr * layer.d_Wi
            Wo_updates = -self.current_lr * layer.d_Wo
            Wh_updates = -self.current_lr * layer.d_Wh
            bias_updates = -self.current_lr * layer.d_bias
        
        layer.Wi += Wi_updates
        layer.Wo += Wo_updates
        layer.Wh += Wh_updates
        layer.bias += bias_updates
    
    def post_update_params(self):
        self.iterations += 1

    




