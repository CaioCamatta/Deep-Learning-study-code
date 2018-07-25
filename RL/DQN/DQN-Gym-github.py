"""
Double Deep Q Network Implementation.
Based on: https://arxiv.org/abs/1509.06461
Set up to work on OpenAI Gym environments with states represented by 1-D arrays of observation
variables and discrete actions such as Classic control, Box2D, etc.
"""

import numpy as np

import tensorflow as tf  # used in custom huber_loss function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras import backend as K


def huber_loss(y_true, y_pred, clip_val=1.0):
    """
    Huber loss implementation based on:
    https://github.com/matthiasplappert/keras-rl/blob/master/rl/util.py
    """
    x = y_true - y_pred
    condition = K.abs(x) < clip_val
    squared_loss = 0.5 * K.square(x)
    linear_loss = clip_val * (K.abs(x) - 0.5 * clip_val)
    return(tf.where(condition, squared_loss, linear_loss))


class ConvDDQN:
    """
    """
    def __init__(self, gamma=0.99, learning_rate=0.001, training_batch_size=32, num_actions=None):
        """
        Inits learner class
        Args (where not clear from name):
            hidden_model_arch: a tuple w/hidden layer sizes for neural net, input and output sizes
            are set by num_input_vars and num_actions
        """

        assert num_actions is not None, "Number of actions is not set"

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.training_batch_size = training_batch_size
        self.num_actions = num_actions
        self.primary_model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), activation='relu', padding='same', input_shape=(16, 16, 9)))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))

        # model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dense(200, activation='relu'))

        model.add(Dense(self.num_actions, activation='linear'))
        adam = Adam(lr=self.learning_rate)
        model.compile(loss=huber_loss, optimizer=adam)
        return(model)

    def train(self, training_batch, return_training_loss=False):

        states, actions, rewards, next_states, dones = training_batch

        y = self.primary_model.predict(states)
        next_q = self.primary_model.predict(next_states)
        next_target_q = self.target_model.predict(next_states)

        idx = np.arange(y.shape[0])
        y[idx, actions] = rewards + (
            np.logical_not(dones) * self.gamma * next_target_q[idx, np.argmax(next_q, axis=1)])

        training_loss = self.primary_model.train_on_batch(states, y)
        if return_training_loss:
            return(training_loss)

    def update_target(self):
        self.target_model.set_weights(self.primary_model.get_weights())
