import numpy as np

def init(input_size, hidden_layer_sizes, output_size):
    global network_0_weight, network_0_bias, network_1, network_2_weight, network_2_bias, network_3, network_4_weight, network_4_bias
    network_0_weight = np.zeros((32768, input_size), dtype=np.float32)
    network_0_bias = np.zeros((32768,), dtype=np.float32) if True else np.zeros((32768,), dtype=np.float32)
    network_1 = None
    network_2_weight = np.zeros((32768, input_size), dtype=np.float32)
    network_2_bias = np.zeros((32768,), dtype=np.float32) if True else np.zeros((32768,), dtype=np.float32)
    network_3 = None
    network_4_weight = np.zeros((output_size, 32768), dtype=np.float32)
    network_4_bias = np.zeros((output_size,), dtype=np.float32) if True else np.zeros((output_size,), dtype=np.float32)

def forward(x, input_size, hidden_layer_sizes, output_size):
    return ((np.maximum(((np.maximum(((x) @ network_0_weight.T + network_0_bias), 0)) @ network_2_weight.T + network_2_bias), 0)) @ network_4_weight.T + network_4_bias)

