import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.python.keras.saving.saved_model.load import metrics
import numpy as np


class RLAgent:
    def __init__(self, input_shape):
        self.model = Sequential([Input(shape=(input_shape,)),
                                 Dense(input_shape * 2, activation="relu", name="layer1"),
                                 Dense(input_shape * 4, activation="relu", name="layer2"),
                                 # Dense(input_shape * 8, activation="relu", name="layer3"),
                                 # Dense(input_shape * 16, activation="relu", name="layer4"),
                                 # Dense(input_shape * 8, activation="relu", name="layer5"),
                                 # Dense(input_shape * 4, activation="relu", name="layer6"),
                                 Dense(input_shape * 2, activation="relu", name="layer7"),
                                 Dense(input_shape, activation="relu", name="layer8"),
                                 Dense(input_shape // 2, activation="relu", name="layer9"),
                                 Dense(2, activation="linear", name="Output_layer")])

    def compile(self, optimizer, loss, metric):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    def summary(self):
        return self.model.summary()

    def fit(self, features, targets, epochs):
        self.model.fit(features, targets, epochs=epochs)

    def predict(self, state):
        state = np.array(state).reshape(1, -1)
        return self.model.predict(state, verbose=0)[0]