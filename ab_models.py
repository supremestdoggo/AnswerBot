import tensorflow as tf
import numpy as np
import math



def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class NumberAI:
    def __init__(self, hidden_layers=[tf.keras.layers.Dense(64)]):
        inputs = tf.keras.Input(shape=(1,))
        outputs = NumberAI.get_outputs([inputs] + hidden_layers + [tf.keras.layers.Dense(1)])
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

    @staticmethod
    def get_outputs(layers):
        if len(layers) == 1:
            return layers[0]
        else:
            return layers[-1](NumberAI.get_outputs(layers[:-1]))
    
    def train(self, inputs, outputs, epochs=1, count=False):
        for x in range(epochs):
            for pos, i in enumerate(inputs):
                self.model.fit(np.array([sigmoid(i)]), np.array([outputs[pos]]))
                if count is True:
                    print(f"Epoch: {x + 1}/{epochs}\nData index: {pos + 1}/{len(inputs)}")
    
    def predict(self, number):
        return self.model.predict(np.array([sigmoid(number)]))[0][0]