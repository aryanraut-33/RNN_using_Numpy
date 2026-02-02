# Recurrent Neural Network (RNN) from Scratch

This project implements a Recurrent Neural Network (RNN) from the ground up using Python and NumPy. It is designed to demonstrate the core concepts of RNNs, including forward propagation, backpropagation through time (BPTT), and optimization using Stochastic Gradient Descent (SGD) with momentum.

## Project Structure

The core implementation is located in the `5 package` directory.
- `rnn.py`: Contains the main RNN model class, optimizer, activation functions, and helper functions for training and inference.
- `test.py`: Demonstrates how to generate time-series data and train/test the model.

## API Documentation

### Training and Inference Helpers

#### `Run_RNN(X_t, Y_t, Activation, epochs=500, n_neurons=500, lr=1e-5, momentum=0.95, decay=0.01, plot_each=0.50, dt=0)`

This is the main driver function to initialize, train, and visualize the RNN.

**Parameters:**
- `X_t` (np.ndarray): Input time-series data for training.
- `Y_t` (np.ndarray): Target time-series data.
- `Activation` (object): An instance of an activation function class (e.g., `Tanh()`).
- `epochs` (int, default=500): Total number of training iterations.
- `n_neurons` (int, default=500): Number of neurons in the hidden state/layer.
- `lr` (float, default=1e-5): Initial learning rate for the optimizer.
- `momentum` (float, default=0.95): Momentum factor for SGD optimizer to accelerate gradients.
- `decay` (float, default=0.01): Decay factor for learning rate over time.
- `plot_each` (float, default=0.50): Frequency of plotting results. If `< 1` (e.g. 0.5), it interprets as a boolean-like check (internally `if not n % plot_each` might be intended as `plot_each` integer frequency, but current implementation uses modulus). Adjust based on specific implementation logic.
- `dt` (int, default=0): Time lag for autoregressive tasks. If `dt != 0`, the model predicts `Y_t` based on `Y_{t-dt}`.

**Returns:**
- `rnn` (RNN_model): The trained RNN model instance.

#### `Apply_RNN(X_t, rnn)`

Runs inference on new input data using a trained RNN model.

**Parameters:**
- `X_t` (np.ndarray): New input time-series data.
- `rnn` (RNN_model): A trained instance of the `RNN_model` class.

**Returns:**
- `Y_hat` (np.ndarray): The predicted output sequence.

---

### Core Classes

#### `class RNN_model`

The main class implementing the Recurrent Neural Network architecture.

**Constructor:** `__init__(self, n_neurons, Activation)`
- Initializes weights (`Wh`, `Wi`, `Wo`) and bias with random small values.
- `n_neurons` (int): Dimension of the hidden state.
- `Activation` (object): Activation function instance.

**Methods:**

- **`forward(self, X_t)`**
  Performs the forward pass over the entire sequence `X_t`.
  - Initializes hidden states `H` and gradients.
  - Calls `RNN_Cell` iteratively for each timestep.
  - Stores `Act_list`, `H`, and `Y_hat` for use in backpropagation.

- **`backward(self, d_values)`**
  Performs Backpropagation Through Time (BPTT).
  - `d_values`: Gradients of the loss function with respect to the outputs.
  - Computes gradients for weights (`d_Wi`, `d_Wh`, `d_Wo`) and biases.
  - Propagates gradients backward through time, accounting for the chain rule across timesteps.

- **`RNN_Cell(self, X_t, h_t, Act_list, H, Y_hat)`**
  Computes the output for a single pass (or loop over sequence) of the RNN cell.
  - **Logic**: $h_t = \text{act}(W_h h_{t-1} + W_i x_t + b)$
  - **Logic**: $\hat{y}_t = W_o h_t$
  - Updates and returns the history lists `Act_list`, `H`, and `Y_hat`.

#### `class SGD_Optimizer`

Implements Stochastic Gradient Descent with Support for Momentum and Learning Rate Decay.

**Constructor:** `__init__(self, lr=1e-5, momentum=0, decay=0)`
- `lr`: Learning Rate.
- `momentum`: Momentum factor (0 to 1).
- `decay`: Learning rate decay factor.

**Methods:**

- **`pre_update_params(self)`**
  Adjusts the current learning rate based on the decay schedule before parameter updates.

- **`update_params(self, layer)`**
  Updates the weights of the given `layer` (the RNN model) using calculated gradients.
  - Applies momentum correction if `momentum > 0`.
  - Updates: `Wi`, `Wo`, `Wh`, `bias`.

- **`post_update_params(self)`**
  Increments the iteration counter, used for learning rate decay calculations.

#### `class Tanh`

Implements the Hyperbolic Tangent activation function.

**Methods:**
- **`forward(self, inputs)`**: Computes $tanh(x)$ and stores inputs.
- **`backward(self, d_values)`**: Computes gradients. Derivative is $1 - tanh^2(x)$.

---

## Interactive Demo (Streamlit App)
We have added a visual, interactive dashboard to help you understand the RNN lifecycle step-by-step.

### Live Demo
**[Click here to try the Interactive App](https://aryanraut-33-rnn-using-numpy-5packageapp-a3wfyz.streamlit.app/)**


### Step-by-Step Guide
The app is divided into 3 tabs:

1.  **Training (Regression)**
    *   What to do: keeping parameters default (`Neurons=50`, `Epochs=500`), click **Start Training**.
    *   Observe: Watch the Red line (Model) learn to fit the Blue line (True Data).
    *   Goal: See how the Loss decreases over time.

2.  **Inference (Prediction)**
    *   What to do: After training in Tab 1, switch to this tab.
    *   Action: Click **Run Apply_RNN() Forward Pass**.
    *   Result: See the model predict on *unseen* data (Green line) using the frozen weights.

3.  **Autoregression (Time Forecasting)**
    *   Theory: Predicting the next value based on past values (Time Lag `dt`).
    *   Experiment: Try `dt=20` (Short memory) vs `dt=50` (Long memory).
    *   Result: See how the model forecasts the future (Red line) beyond the training data.
