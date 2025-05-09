import numpy as np

class NeuralNet:
    def __init__(self):
        # Seed the random number generator for reproducibility
        np.random.seed(1)
        
        # Initialize synaptic weights (3 inputs → 1 output)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def __sigmoid(self, x):
        # Activation function
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        # Derivative for backpropagation
        return x * (1 - x)

    def train(self, inputs, outputs, training_iterations):
        for _ in range(training_iterations):
            output = self.learn(inputs)
            error = outputs - output
            adjustments = np.dot(inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def learn(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))


if __name__ == "__main__":
    # Initialize the neural network
    neural_network = NeuralNet()

    # Input dataset (3 training examples, 3 input features)
    inputs = np.array([[0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 1]])

    # Output dataset (labels)
    outputs = np.array([[1], [0], [1]])

    # Train the neural network
    neural_network.train(inputs, outputs, training_iterations=10000)

    # Test with a new input
    test_input = np.array([1, 0, 1])
    result = neural_network.learn(test_input)

    print("Test Output:", result)
