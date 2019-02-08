import torch
from neural_network import NeuralNetwork


class AND():

    def __init__(self):
        # Appropriately initialize your neural network
        self.model = NeuralNetwork([2, 1])

        # Set weights of model
        # w0 = 3, w1 = 3, b0 = 5
        weights = NeuralNetwork.getLayer(self.model, 0)
        weights[0], weights[1] = 3, 3
        bias = self.model.biasMatrix[0]
        bias[0] = 5

    def __call__(self, x1, x2):
        # Process boolean inputs and send them to your neural network for a forward pass
        inputs = torch.Tensor([[float(x1), float(x2)]])
        return self.model.forward(inputs)


class OR():

    def __init__(self):
        # Appropriately initialize your neural network
        self.model = NeuralNetwork([2, 1])

        # Set weights of model
        # w0 = 3, w1 = 3, b0 = 1
        weights = NeuralNetwork.getLayer(self.model, 0)
        weights[0], weights[1] = 3, 3
        bias = self.model.biasMatrix[0]
        bias[0] = 1

    def __call__(self, x1, x2):
        # Process boolean inputs and send them to your neural network for a forward pass
        inputs = torch.Tensor([[float(x1), float(x2)]])
        return self.model.forward(inputs)


class NOT():

    def __init__(self):
        # Appropriately initialize your neural network
        self.model = NeuralNetwork([1, 1])

        # Set weights of model
        # w0 = -2, b0 = -1
        weights = NeuralNetwork.getLayer(self.model, 0)
        weights[0] = -2
        bias = self.model.biasMatrix[0]
        bias[0] = -1

    def __call__(self, x1):
        # Process boolean inputs and send them to your neural network for a forward pass
        inputs = torch.Tensor([[float(x1)]])
        return self.model.forward(inputs)


class XOR():

    def __init__(self):
        # Appropriately initialize your neural network
        self.model = NeuralNetwork([2, 2, 1])

        # Set weights of model

        # Hidden Layer
        # w0 = -2, b0 = -1
        weights = NeuralNetwork.getLayer(self.model, 0)
        weights[0][0], weights[0][1] = 1, -1
        weights[1][0], weights[1][1] = -1, 1
        bias = self.model.biasMatrix[0]
        bias[0] = 1

        # Output Layer
        weights = NeuralNetwork.getLayer(self.model, 1)
        weights[0], weights[1] = 1, 1
        bias = self.model.biasMatrix[1]
        bias[0] = -0.1

    def __call__(self, x1, x2):
        # process boolean inputs and send them to your neural network for a forward pass
        # Convert booleans to x1, x2
        if not x1:
            x1 = -1
        if not x2:
            x2 = -1

        # Forward Pass
        inputs = torch.Tensor([[float(x1), float(x2)]])
        return self.model.forward(inputs)
