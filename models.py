import tensorflow as tf
import numpy as np
import math


class NumberAI:
    def __init__(self, hidden_layers=[64]):
        inputs = tf.keras.Input(shape=(1,))
        outputs = NumberAI.get_outputs(hidden_layers + [tf.keras.layers.Dense(1)])
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    def get_outputs(layers):
        if len(layers) == 1:
            return layers[0]
        else:
            return layers[-1](NumberAI.get_outputs(layers[:-1]))
    
    def train(self, inputs, outputs):
        for i, pos in enumerate(inputs):
            self.model.fit(np.array([NumberAI.sigmoid(i)]), np.array(outputs[pos]))
    
    def predict(self, number):
        return self.model.predict(np.array([NumberAI.sigmoid(number)]))[0][0]