# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import copy

# ==========================================
# 1. Activation Classes
# ==========================================
### Defining activation [tanh] and its derivative

class Tanh():

    def forward(self, inputs):
        self.output = np.tanh(inputs)
        self.inputs = inputs
    
    def backward(self, d_values):
        derivative = 1 - self.output**2
        self.d_inputs = d_values * derivative

# ==========================================
# 2. Optimizer
# ==========================================

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

# ==========================================
# 3. Core RNN Model
# ==========================================

'''
x_t -> input at a timestep t : [1,1]
y_t -> output at t : [1,1] or [1,C] where C = vocab size (possible no of outcomes), in case of textual prediction
h_t -> hidden state at t : [n_neurons,1]

W_i -> weights corresponding to inputs : [n_neurons,1]
W_o -> weights corresponding to the output at t : [1,n_neurons]
W_h -> weights corresponding to the hidden state at t : [n_neurons,n_neurons]
'''
class RNN_model():

    def __init__(self, n_neurons, Activation):


        self.n_neurons = n_neurons

        # set up the matrices for the learnable parameters
        self.Wh = 0.1*np.random.randn(n_neurons,n_neurons)
        self.Wi = 0.1*np.random.randn(n_neurons,1)
        self.Wo = 0.1*np.random.randn(1,n_neurons)
        self.bias = 0.1*np.random.randn(n_neurons,1)

        self.Activation = Activation
    
    def forward(self, X_t):
        
        self.T = max(X_t.shape)
        self.X_t = X_t
        
        self.Y_hat = np.zeros((self.T, 1)) 
        self.H = [np.zeros((self. n_neurons,1)) for t in range(self.T+1)]

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

        Act_list = [copy.copy(Activation) for t in range(self.T)]    

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


# ==========================================
# 4. User-Facing Functions (Train/Predict)
# ==========================================

def Run_RNN(X_t, Y_t, Activation, epochs = 500, n_neurons = 500,  lr = 0.00001, momentum = 0.95, decay = 0.01, plot_each = 0.50, dt = 0):

    '''
    The paraemeters of the function Run_RNN (training model: both x and y should be available):
         1. X_t: Input sequential data or features for training; typically used when dt=0.
         2. Y_t: Target sequential data; also used as input (shifted) if dt != 0.
         3. Activation: The activation function object (e.g., Tanh()) to be used in the hidden layers. [We have defined only Tanh() activation here]
         4. epochs: Total number of training iterations to perform. (default = 500)
         5. n_neurons: Number of neurons in the hidden layer of the RNN. (default = 500)
         6. lr: Initial learning rate for the optimizer. (default = 0.00001: such low as higher value may suffer from exploding gradient problem)
         7. momentum: Parameter that accelerates SGD in the relevant direction and dampens oscillations. (default = 0.95)
         8. decay: Factor by which the learning rate decays over time. (default = 0.01)
         9. plot_each: Frequency of plotting the results (e.g., every Nth epoch); if < 1, plots every epoch. (default = 1), i.e plot for every epoch
         10. dt: Time step lag for autoregressive tasks; if dt != 0, model predicts Y_t based on Y_{t-dt}.
    '''
     
    rnn = RNN_model(n_neurons, Activation)
    T = max(X_t.shape)
    
    if dt >= T:
        raise ValueError(f"Time lag dt={dt} must be smaller than the data size T={T}. Please increase data size or reduce dt.")
        
    #epochs = 200
    #lr = 0.00001
    optimizer = SGD_Optimizer(lr, momentum, decay)
    
    X_plot = np.arange(0, T)
    
    if dt != 0 :
        X_t_dt = Y_t[:-dt]
        Y_t_dt = Y_t[dt : ]
        X_plots = X_plot[dt:]
        
    else:
        X_t_dt = X_t
        Y_t_dt = Y_t
        X_plots = X_plot
        

    #Monitor = np.zeros((epochs,1))
    
    print("RNN running succesfully...")

    for n in range(epochs):
        

        rnn.forward(X_t_dt)
        
        dY = rnn.Y_hat - Y_t_dt
          
        #Monitor[n] = Loss
        
        rnn.backward(dY)
        
        optimizer.pre_update_params()
        optimizer.update_params(rnn)
        optimizer.post_update_params() 
        
        if not n % plot_each:
            rnn.forward(X_t)
            
            m = np.min(np.vstack((rnn.Y_hat, Y_t)))
            M = np.max(np.vstack((rnn.Y_hat, Y_t)))

            Loss = 0.5 *np.dot(dY.T,dY)/(T-dt)
            
            
            plt.plot(X_plot, Y_t)
            plt.plot(X_plot+ dt, rnn.Y_hat)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(['y', '$\hat{y}$'])
            plt.title('epoch# ' + str(n))
            if dt != 0:
                plt.fill_between([X_plot[-1], X_plots[-1] + dt],m, M, color = 'k', alpha = 0.1)
                plt.plot([X_plot[-1], X_plot[-1]], [m, M],'k-',linewidth = 3)
            plt.show()
            
            Loss = float(Loss)
            print(f'Current MSE at epoch {n}: + {Loss:.3f}' )
    
    rnn.forward(X_t)
    
    if dt != 0  :
        dy = rnn.Y_hat[:-dt] - Y_t[dt:]
    else:
        dy = rnn.Y_hat - Y_t
   
   
    Loss = 0.5 *np.dot(dY.T,dY)/(T-dt)
    
    plt.plot(X_plot, Y_t)
    plt.plot(X_plot+ dt, rnn.Y_hat)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['y', '$\hat{y}$'])
    plt.title('epoch# ' + str(n))
    if dt != 0:
        plt.fill_between([X_plot[-1], X_plots[-1] + dt],m, M, color = 'k', alpha = 0.1)
        plt.plot([X_plot[-1], X_plot[-1]], [m, M],'k-',linewidth = 3)
    plt.show()

        
    Loss = float(Loss)
    print(f'Model training complete! \nLoss = {Loss: .3f}')
    
    return rnn

##############

def Apply_RNN(X_t, rnn):
    T = max(X_t.shape)
    Y_hat = np.zeros((T,1))
     
    # Initialize a new hidden state for this sequence
    H = [np.zeros((rnn.n_neurons,1)) for t in range(T+1)]
    ht = H[0]  # Initial hidden state (all zeros)
    
    Act_list = [rnn.Activation for t in range(T)]
    
    [_,_,Y_hat] = rnn.RNN_Cell(X_t, ht, Act_list, H, Y_hat)
    
    return Y_hat
#############
