import tensorflow as tf
import numpy as np
import math



def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def numpy_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.divide(1, 1 + np.exp(np.multiply(-1, x)))

def get_outputs(layers):
    if len(layers) == 1:
        return layers[0]
    else:
        return layers[-1](get_outputs(layers[:-1]))

class NumberAI:
    """Simple class for AI with single-number I/O."""
    def __init__(self, hidden_layers=[tf.keras.layers.Dense(64), tf.keras.layers.Dense(64)]):
        inputs = tf.keras.Input(shape=(1,))
        outputs = NumberAI.get_outputs([inputs] + hidden_layers + [tf.keras.layers.Dense(1)])
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    
    def train(self, inputs, outputs, epochs=1):
        sigmoid_inputs = [np.array([sigmoid(x)]) for x in inputs]
        array_outputs = [np.array([x]) for x in outputs]
        training_data = tf.data.Dataset.from_tensor_slices((sigmoid_inputs, array_outputs))
        self.model.fit(training_data, epochs=epochs)
    
    def predict(self, number):
        return self.model.predict(np.array([sigmoid(number)]))[0][0]


class ArrayAI:
    """Simple class for AI with ndarray I/O. Currently in development and should not be used."""
    def __init__(self, input_shape, output_shape, hidden_layers=[tf.keras.layers.Dense(64), tf.keras.layers.Dense(64)]):
        self.output_shape = output_shape
        self.flat_output_shape = (math.prod(output_shape),)
        inputs = tf.keras.Input(shape=input_shape)
        outputs = get_outputs([inputs] + hidden_layers + [tf.keras.layers.Dense(self.flat_output_shape[0])])
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile()
    
    def train(self, input_arrays, output_arrays, epochs=1):
        sigmoid_inputs = [numpy_sigmoid(x) for x in input_arrays]
        flat_outputs = [np.reshape(x, self.flat_output_shape) for x in output_arrays]
        training_data = tf.data.Dataset.from_tensor_slices((sigmoid_inputs, flat_outputs))
        self.model.fit(training_data, epochs=epochs)
    
    def predict(self, input_array):
        flat_prediction = self.model.predict(numpy_sigmoid(input_array))
        return np.reshape(flat_prediction, self.output_shape)