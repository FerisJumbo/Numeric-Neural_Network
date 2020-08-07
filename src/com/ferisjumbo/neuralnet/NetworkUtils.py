import numpy as np


class Layer_Dense:
    def __init__(
        self,
        inputs,
        neurons,
        weight_regularizer_l1=0,
        weight_regularizer_l2=0,
        bias_regularizer_l1=0,
        bias_regularizer_l2=0,
    ):
        self.weights = 0.1 * np.random.randn(inputs, neurons)
        self.biases = np.zeros(shape=(1, neurons))

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward_propagation(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward_propagation(self, dvalues):
        self.dweights = np.dot(np.array(self.inputs).T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Punishes the Network for low weight values
        if self.weight_regularizer_l1 > 0:
            dL1 = self.weights.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # Punishes the Network for high weight values
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # Punishes the Network for low bias values
        if self.weight_regularizer_l1 > 0:
            dL1 = self.biases.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # Punishes the Network for high bias values
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dvalues = np.dot(dvalues, np.array(self.weights).T)


class ReLU_Activation:
    def forward_propagation(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward_propagation(self, dvalues):
        self.dvalues = dvalues.copy()
        self.dvalues[self.inputs <= 0] = 0


class Softmax_Activation:
    def forward_propagation(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward_propagation(self, dvalues):
        self.dvalues = dvalues.copy()


class Loss:
    def regularization_loss(self, layer):
        regul_loss = 0

        if layer.weight_regularizer_l1 > 0:
            regul_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2 > 0:
            regul_loss += layer.weight_regularizer_l2 * np.sum(
                layer.weights * layer.weights
            )

        if layer.bias_regularizer_l1 > 0:
            regul_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 > 0:
            regul_loss += layer.bias_regularizer_l2 * np.sum(
                layer.biases * layer.biases
            )

        return regul_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward_propagation(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred = y_pred[range(samples), y_true]
        negative_log_likelihoods = -np.log(y_pred)
        data_loss = np.mean(negative_log_likelihoods)
        return data_loss

    def backward_propagation(self, dvalues, y_true):
        samples = dvalues.shape[0]
        self.dvalues = dvalues.copy()
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues = self.dvalues / samples


def normalize_data(data):
    data = np.array(data)
    normalized_data = data / np.full(data.shape, 255)
    return normalized_data.tolist()
