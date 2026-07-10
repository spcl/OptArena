import numpy as np

def init(input_size, hidden_size, output_size):
    global hidden, i2h_weight, i2h_bias, h2o_weight, h2o_bias, tanh
    hidden = np.zeros((batch_size, hidden_size), dtype=np.float32)
    i2h_weight = np.zeros((hidden_size, input_size + hidden_size), dtype=np.float32)
    i2h_bias = np.zeros((hidden_size,), dtype=np.float32) if True else np.zeros((hidden_size,), dtype=np.float32)
    h2o_weight = np.zeros((output_size, hidden_size), dtype=np.float32)
    h2o_bias = np.zeros((output_size,), dtype=np.float32) if True else np.zeros((output_size,), dtype=np.float32)
    tanh = None

def forward(x, initial_hidden, input_size, hidden_size, output_size):
    if (initial_hidden is not None):
        hidden = initial_hidden
    hidden = hidden
    combined = np.concatenate((x, hidden), axis=1)
    hidden = np.tanh(((combined) @ i2h_weight.T + i2h_bias))
    output = ((hidden) @ h2o_weight.T + h2o_bias)
    return output

