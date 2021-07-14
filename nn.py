import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):

        # TODO
        # layer_sizes example: [4, 10, 2]
        self.layer_sizes = layer_sizes
        # weights
        w1 = np.random.randn(layer_sizes[1], layer_sizes[0])
        w2 = np.random.randn(layer_sizes[2], layer_sizes[1])
        self.weights = [w1, w2]
        # biases
        b1 = np.zeros((layer_sizes[1], layer_sizes[0]))
        b2 = np.zeros((layer_sizes[2], layer_sizes[1]))
        self.biases = [b1, b2]

    def activation(self, x):
        # TODO
        # return x
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        
        # TODO
        # x example: np.array([[0.1], [0.2], [0.3]])
        z1 = (self.weights[0] @ x) + self.biases[0]
        a1 = np.asarray([self.activation(z[0]) for z in z1]).reshape((self.layer_sizes[1], 1))
        z2 = (self.weights[1] @ a1) + self.biases[1]
        a2 = np.asarray([self.activation(z[0]) for z in z2]).reshape((self.layer_sizes[2], 1))
        print(a2)


if __name__ == '__main__':
    nn = NeuralNetwork([4, 10, 2])
    nn.forward(np.array([[0.1], [0.2], [0.3], [0.4]]))
