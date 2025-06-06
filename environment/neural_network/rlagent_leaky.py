import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.saving.saved_model.load import metrics
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
import numpy as np


class RLAgent:
    def __init__(self, sequence_length, input_shape, action_output=2, learning_rate=0.95):
        self.input_shape = (sequence_length, input_shape)
        self.action_output = action_output
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def _build_model(self):
        return Sequential([
            Input(shape=self.input_shape),
            Flatten(),
            Dense(64), LeakyReLU(alpha=0.01),
            Dense(128), LeakyReLU(alpha=0.01),
            Dense(256), LeakyReLU(alpha=0.01),
            Dense(128), LeakyReLU(alpha=0.01),
            Dense(64), LeakyReLU(alpha=0.01),
            Dense(self.action_output, activation="linear", name="output_layer")
        ])

    @tf.function
    def _predict_fast(self, state):
        # DROPOUT WHEN TRAINING = TRUE
        return self.model(state, training=True)

    def predict(self, state):
        state = np.array(state, dtype=np.float32)  # Force batch dim
        output = self._predict_fast(state)
        return output.numpy()[0]

    @tf.function
    def _train_step(self, obs_batch, action_batch, returns):
        # Calculate advantages from baseline
        advantages = returns - tf.reduce_mean(returns)

        with tf.GradientTape() as tape:
            predictions = self.model(obs_batch, training=True)

            # MSE between predicted and taken action, weighted by advantage
            loss_elements = tf.reduce_sum(tf.square(predictions - action_batch), axis=1)
            loss = tf.reduce_mean(loss_elements * advantages)

        # Compute and apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, obs_batch, action_batch, returns):
        obs_batch = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        returns = (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8)

        loss = self._train_step(obs_batch, action_batch, returns)
        return float(loss.numpy())

    def save(self, path, include_optimizer=True):
        self.model.save(path, include_optimizer=include_optimizer)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)