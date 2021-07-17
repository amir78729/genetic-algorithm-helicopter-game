import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):

        self.w0 = np.random.randn(layer_sizes[1], layer_sizes[0])
        self.b0 = np.random.randn(layer_sizes[1], 1)
        self.hidden_layer = np.zeros((layer_sizes[1], 1))
        self.w1 = np.random.randn(layer_sizes[2], layer_sizes[1])
        self.b1 = np.random.randn(layer_sizes[2], 1)
        self.output_layer = np.zeros((layer_sizes[2], 1))

        # layer_sizes example: [4, 10, 2]

    def activation(self, x):
        z = 1/(1 + np.exp(-x))
        # x = np.maximum(0, x)
        return z
        # return np.maximum(x, 0)
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[1]):
        #         x[i][j] = max(0, x)
        # return x

    def forward(self, x):
        self.hidden_layer = self.activation((self.w0 @ x) + self.b0)
        self.output_layer = self.activation((self.w1 @ self.hidden_layer) + self.b1)
        # x example: np.array([[0.1], [0.2], [0.3]])


# n = NeuralNetwork([6, 20, 1])
# n.forward(np.array([[0.1], [0.2], [0.3], [0.4], [0.7], [0.8]]))
# print(n.output_layer)
