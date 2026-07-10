import numpy as np

def init(input_size, hidden_size, output_size):
    global i2h_weight, i2h_bias, h2o_weight, h2o_bias, tanh
    i2h_weight = np.zeros((hidden_size, input_size + hidden_size), dtype=np.float32)
    i2h_bias = np.zeros((hidden_size,), dtype=np.float32) if True else np.zeros((hidden_size,), dtype=np.float32)
    h2o_weight = np.zeros((output_size, hidden_size), dtype=np.float32)
    h2o_bias = np.zeros((output_size,), dtype=np.float32) if True else np.zeros((output_size,), dtype=np.float32)
    tanh = None

def forward(x, h0, input_size, hidden_size, output_size):
    (seq_len, batch_size, _) = x.shape
    hidden = h0
    outputs = []
    for t in range(seq_len):
        combined = np.concatenate((x[t], hidden), axis=1)
        hidden = np.tanh(((combined) @ i2h_weight.T + i2h_bias))
        output = ((hidden) @ h2o_weight.T + h2o_bias)
        outputs.append(output)
    return np.stack(outputs, axis=0)

