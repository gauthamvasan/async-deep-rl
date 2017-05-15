#!/usr/bin/python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disables debuugging logs of tensorflow

import unittest
from atari_wrapper import  Atari_Environment
import gym
import tensorflow as tf
from build_model import sequential_network
import numpy as np
from keras import backend as K

class Atari_Tests(unittest.TestCase):
    def test(self):
        env = Atari_Environment(gym.make('SpaceInvaders-v0'), scaled_width=84, scaled_height = 84, last_m_frames=4)
        start_state = env.reset()
        self.assertEqual(start_state.shape, (4,84,84))

        state, reward, done, info = env.step(1)
        self.assertEqual(state.shape, (4, 84, 84))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

class Model_Tests(unittest.TestCase):
    def test(self):
        g = tf.Graph()
        session = tf.Session()
        K.set_session(session)
        state = tf.placeholder("float",
                               [None, 4, 84, 84])
        q_network = sequential_network(num_actions = 6, history_length = 4, scaled_width = 84, scaled_height = 84)
        network_params = q_network.trainable_weights
        q_values = q_network(state)

        st = np.zeros((4,84,84))
        session.run(tf.global_variables_initializer())

        q_vals = q_values.eval(session=session, feed_dict={state: [st]})
        self.assertEqual(q_vals.shape,(1,6))
