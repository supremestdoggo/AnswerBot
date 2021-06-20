import tensorflow as tf
import numpy as np
import math



def sigmoid(x):
    return 1 / (1 + math.exp(-x))

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
    
    def train(self, inputs, outputs, epochs=1, **kwargs):
        # self.model.fit(np.array([sigmoid(i)]), np.array([outputs[pos]]))
        sigmoid_inputs = [np.array([sigmoid(x)]) for x in inputs]
        array_outputs = [np.array([x]) for x in outputs]
        train_data = tf.data.Dataset.from_tensor_slices((sigmoid_inputs, array_outputs))
        self.model.fit(train_data, epochs=epochs, **kwargs)

    
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