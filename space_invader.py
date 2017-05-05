import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Disables debuugging logs of tensorflow
import numpy as np
import math
import gym
import cv2
from atari_wrapper import Atari_Environment
from build_model import build_network, sequential_network
from keras import backend as K
import tensorflow as tf


def initialize_graph_ops(num_actions):
    state = tf.placeholder("float", [None, 4, 84, 84])
    #q_network = build_network(6)
    q_network = sequential_network(6)
    network_params = q_network.trainable_weights
    q_values = q_network(state)

    graph_ops = {"q_values": q_values,
                 "state": state}
    return graph_ops


def train(session, env, graph_ops):
    q_values = graph_ops["q_values"]
    session.run(tf.initialize_all_variables())
    state = graph_ops["state"]

    for i_episode in range(2):
        observation = env.reset()
        #st = np.stack([observation],axis=0)
        #print st.shape

        print q_values.eval(session=session, feed_dict = {state: [observation]})
        print observation.shape
        for t in range(100):
            env.render()
            action = np.random.choice(range(env.num_actions))
            observation, reward, done, info = goose.step(action)
            #print (reward)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

if __name__ == "__main__":
    g = tf.Graph()
    with g.as_default(), tf.Session() as session:
        K.set_session(session)
        num_actions = 6
        graph_ops = initialize_graph_ops(num_actions)

        env = gym.make('SpaceInvaders-v0')
        goose = Atari_Environment(env)

        train(session, goose, graph_ops)


