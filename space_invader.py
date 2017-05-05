#!/usr/bin/python
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

flags = {}

# Choose the Gym environment you intend to run
flags['experiment'] = 'async_dqn_space_invader' # Name of the experiment
flags['game'] = 'SpaceInvaders-v0' # OpenAI Gym handle/name for the game

# Algorithm specific flags and hyper-parameters
flags['num_actor_threads'] = 8 # Number of concurrent actor-learner threads to use during training.
flags['T_max'] = 80000000      # Number of training frames/steps
flags['async_update_frequency'] = 5 # Frequency with which each actor learner thread does an async gradient update
flags['target_network_update_frequency'] = 40000 # Update and Reset the target network every n timesteps
flags['learning_rate'] = 0.0001   # Initial learning rate
flags['gamma'] = 0.99 # Discount rate for the reward
flags['anneal_epsilon_timesteps'] = 4000000 # 'Number of timesteps to anneal epsilon.

# Pre-processing parameters (The RGB image is pre-processed to fit computational requirements)
flags['scaled_width'] =  84    # Scale screen to this width.
flags['scaled_height'] = 84    # Scale screen to this height.
flags['agent_history_length'] = 4 # Use this number of recent screens as the environment state.

# Summary writer
flags['summary_dir'] = '~/Downloads/tmp/summaries' # Directory for storing tensorboard summaries
flags['checkpoint_dir'] = '~/Downloads/tmp/checkpoints' # Directory for storing model checkpoints
flags['summary_interval'] = 5 # Save training summary to file every n seconds (rounded up to statistics interval)
flags['checkpoint_interval'] = 600 # Checkpoint the model (i.e. save the parameters) every n seconds (rounded up to statistics interval.)
flags['checkpoint_path'] = 'path/to/recent.ckpt' # Path to recent checkpoint to use for evaluation
flags['eval_dir'] = '~/Downloads/tmp/' # Directory to store gym evaluation

# Testing & Rendering
flags['render_training'] = True, # If true, have gym render environments during training
flags['testing'] = False # If true, run gym evaluation

flags['num_eval_episodes'] = 100 # Number of episodes to run gym evaluation

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


