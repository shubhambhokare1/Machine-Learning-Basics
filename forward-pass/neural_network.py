# Import Necessary Modules

import torch
import numpy


# Neural Network API

class NeuralNetwork():

    def __init__(self, sizes):
        # Declare a List that stores all the Tensor Matrices
        # for mappings from Layer to Layer + 1
        self.weightMatrix = list()
        self.biasMatrix = list()

        for layer in range(len(sizes) - 1):
            # Populate a tensor with random values with conditions
            # Mean = 0, Standard Deviation = 1 / sqrt(layer_size)
            input_size = sizes[layer]
            output_size = sizes[layer + 1]

            # Initialize weights
            weights = torch.zeros((input_size, output_size))
            torch.Tensor.normal_(weights, mean=0, std=(1 / pow(input_size, 0.5)))

            # Initialize biases
            bias = torch.zeros((1, output_size))

            # Append to final weight matrix and bias collection
            self.weightMatrix.append(weights)
            self.biasMatrix.append(bias)

    def getLayer(self, index):
        # Return requested weights
        return self.weightMatrix[index]

    def forward(self, x):
        # Perform neural network forward pass
        for layer in range(len(self.weightMatrix)):
            weights = self.weightMatrix[layer]
            bias = self.biasMatrix[layer]

            # Perform matrix multiplication
            # Summation (inputs * weights) - bias
            layer_input = x
            layer_output = torch.Tensor.addmm(-bias, layer_input, weights)
            # Apply activation function - sign function
            x = numpy.sign(layer_output)

        # Return appropriate value based on x
        if x[0] > 0:
            return True
        else:
            return False
