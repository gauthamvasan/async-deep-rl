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
import threading, time

T = 0   # Global counter
flags = {}

# Choose the Gym environment you intend to run
flags['experiment'] = 'async_dqn_1step_space_invader' # Name of the experiment
flags['game'] = 'SpaceInvaders-v0' # OpenAI Gym handle/name for the game

# Algorithm specific flags and hyper-parameters
flags['num_actor_threads'] = 8 # Number of concurrent actor-learner threads to use during training.
flags['T_max'] = 80000000      # Number of training frames/steps
flags['async_update_frequency'] = 5 # Frequency with which each actor learner thread does an async gradient update
flags['target_network_update_frequency'] = 40000 # Update and Reset the target network every n timesteps
flags['learning_rate'] = 0.0001   # Initial learning rate
flags['gamma'] = 0.99 # Discount rate for the reward

# Pre-processing parameters (The RGB image is pre-processed to fit computational requirements)
flags['scaled_width'] =  84    # Scale screen to this width.
flags['scaled_height'] = 84    # Scale screen to this height.
flags['agent_history_length'] = 4 # Use this number of recent screens as the environment state.

# Sample final epsilon from the value listed below
flags["final_epsilon_choices"] = [0.1, 0.01, 0.5]
flags["final_epsilon_choice_probabilities"] = [0.4, 0.3, 0.3]
flags['anneal_epsilon_timesteps'] = 4000000 # 'Number of timesteps to anneal epsilon.

# Summary writer
flags['summary_dir'] = '~/Downloads/tmp/summaries' # Directory for storing tensorboard summaries
flags['checkpoint_dir'] = '~/Downloads/tmp/checkpoints' + "/" + flags["experiment"] # Directory for storing model checkpoints
flags['summary_interval'] = 5 # Save training summary to file every n seconds (rounded up to statistics interval)
flags['checkpoint_interval'] = 600 # Checkpoint the model (i.e. save the parameters) every n seconds (rounded up to statistics interval.)
flags['eval_dir'] = '~/Downloads/tmp/' # Directory to store gym evaluation
flags['checkpoint_file'] = "/filename.ckpt"

# Testing & Rendering
flags['render_training'] = False, # If true, have gym render environments during training
flags['testing'] = False # If true, run gym evaluation

flags['num_eval_episodes'] = 100 # Number of episodes to run gym evaluation

def initialize_graph_ops(num_actions):

    # Shared network initializations
    state = tf.placeholder("float",
                           [None, flags['agent_history_length'], flags['scaled_width'], flags['scaled_height']])
    shared_q_network = sequential_network(num_actions)
    network_params = shared_q_network.trainable_weights
    q_values = shared_q_network(state)

    # Target network initializations (for async updates)
    target_state = tf.placeholder("float",
                           [None, flags['agent_history_length'], flags['scaled_width'], flags['scaled_height']])
    target_q_network = sequential_network(num_actions)
    target_network_params = target_q_network.trainable_weights
    target_q_values = target_q_network(target_state)

    # Op for async updates of target network with shared/online network weights
    async_update_target_network = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]

    # Cost and gradient update ops
    y = tf.placeholder("float", [None])                       # Target
    actions_list = tf.placeholder(tf.int32, [None, None])     # 2D list consisting of sample number (in th batch) and the action chosen
    action_values = tf.gather_nd(q_values,actions_list)       # Gather the Q values

    cost = tf.reduce_mean(tf.square(y - action_values))
    optimizer = tf.train.RMSPropOptimizer(flags["learning_rate"])
    gradient_update = optimizer.minimize(cost,var_list=network_params)


    graph_ops = {"q_values": q_values,
                 "state": state,
                 "target_state": target_state,
                 "target_q_values": target_q_values,
                 "async_update_target_network": async_update_target_network,
                 "action_values": action_values,
                 "action_list": actions_list,
                 "y": y,
                 "cost": cost,
                 "optimizer": optimizer,
                 "gradient_update": gradient_update,
                 }


    return graph_ops

def initialize_summary_ops():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Episode Reward", episode_reward)
    avg_episode_max_q = tf.Variable(0.)
    tf.summary.scalar("Average_max Q Value", avg_episode_max_q)
    logged_epsilon = tf.Variable(0.)
    tf.summary.scalar("Epsilon", logged_epsilon)
    episode_timesteps = tf.Variable(0.)
    tf.summary.scalar("Episode_timesteps", episode_timesteps)

    summary_vars = [episode_reward, avg_episode_max_q, logged_epsilon, episode_timesteps]
    summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op

def sample_action_epsilon_greedy(q_values, action_set, epsilon=0.1):
    if np.random.rand() < epsilon:
        action = np.random.choice(action_set)
    else:
        action = np.argmax(q_values)
    return action

def actor_learner(thread_id, env, session, graph_ops, num_actions, summary_ops, saver):
    global T
    t = 0 # Thread step counter
    summary_placeholders, update_ops, summary_op = summary_ops

    # Selecting epsilon as described in the paper
    epsilon = 1.0
    final_epsilon = np.random.choice(flags["final_epsilon_choices"], p=flags["final_epsilon_choice_probabilities"])
    epsilon_anneal_factor = (epsilon - final_epsilon)/flags['anneal_epsilon_timesteps']

    state_mem = []
    action_mem = []
    y_mem = []
    action_set = np.arange(num_actions)
    batch_counter = 0

    while T < flags["T_max"]:
        current_state = env.reset()
        done = False

        episode_return = 0.0
        avg_episode_max_q = 0.0
        episode_timesteps = 0

        while not done:
            q_values = graph_ops["q_values"].eval(session=session, feed_dict={graph_ops["state"]: [current_state]})
            action = sample_action_epsilon_greedy(q_values, action_set, epsilon)

            next_state, reward, done, info = env.step(action=action)
            next_q_values = graph_ops["target_q_values"].eval(session=session,
                                                              feed_dict={graph_ops["target_state"]: [next_state]})
            scaled_reward = np.clip(reward,-1,1)

            if done:
                y_mem.append(scaled_reward)
                # Print statistics of the agent's performance this episode
                stats = [episode_return+reward, (avg_episode_max_q+np.max(q_values))/float(episode_timesteps), epsilon, episode_timesteps]
                for i in range(len(stats)):
                    session.run(update_ops[i], feed_dict={summary_placeholders[i]: float(stats[i])})
                print "THREAD:", thread_id, "/ TIME", T, "/ TIMESTEP", t, "/ EPSILON", epsilon, "/ REWARD", episode_return, "/ Q_MAX %.4f" % (
                avg_episode_max_q/ float(episode_timesteps)), "/ EPSILON PROGRESS", t / float(flags["anneal_epsilon_timesteps"])
            else:
                y_mem.append(scaled_reward + flags["gamma"]*np.max(next_q_values))

            # Update memory
            state_mem.append(current_state)
            action_mem.append([t-batch_counter,action])

            # Summary writer info updates
            episode_timesteps += 1
            episode_return += reward
            avg_episode_max_q += np.max(q_values)

            # Update counters
            T += 1
            t += 1

            # Update the state
            current_state = next_state

            # Updates of target network
            if T % flags["target_network_update_frequency"] == 0:
                session.run(graph_ops["async_update_target_network"])

            if t % flags['async_update_frequency'] == 0 or done:
                session.run(graph_ops["gradient_update"], feed_dict = {graph_ops["y"]: y_mem,
                                                                       graph_ops["state"]: state_mem,
                                                                       graph_ops["action_list"]: action_mem})

                # Clear the gradients
                y_mem = []
                action_mem = []
                state_mem = []
                batch_counter = t

            # Save the model every 'n' seconds
            if t % flags['checkpoint_interval'] == 0:
                saver.save(session, flags["checkpoint_dir"])



def train(session, num_actions, graph_ops, saver):
    global T
    session.run(graph_ops["async_update_target_network"])
    envs = [Atari_Environment(gym.make('SpaceInvaders-v0'),flags["scaled_width"],flags["scaled_height"],
                              flags["agent_history_length"]) for i in range(flags["num_actor_threads"])]

    summary_placeholders, update_ops, summary_op = initialize_summary_ops()

    # Initialize tensorflow variables
    session.run(tf.initialize_all_variables())
    summary_save_path = flags["summary_dir"] + "/" + flags["experiment"]
    writer = tf.summary.FileWriter(summary_save_path, session.graph)
    if not os.path.exists(flags["checkpoint_dir"]):
        os.makedirs(flags["checkpoint_dir"])

    # Start num_concurrent actor-learner training threads
    actor_learner_threads = [threading.Thread(target=actor_learner, args=(
    thread_id, envs[thread_id], session, graph_ops, num_actions, [summary_placeholders, update_ops, summary_op], saver))
                             for thread_id in range(flags["num_actor_threads"])]
    for t in actor_learner_threads:
        t.start()

    # Show the agents training and write summary statistics
    last_summary_time = 0
    while T <= flags["T_max"]:
        if flags["render_training"]:
            for env in envs:
                env.render()
        now = time.time()
        if now - last_summary_time > flags["summary_interval"]:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now

    for t in actor_learner_threads:
        t.join()

def evaluation(session, num_actions, graph_ops, saver):
    g = tf.Graph()

    saver.restore(session, flags['checkpoint_dir']+flags['checkpoint_file'])
    print "Restored model weights from ", flags['checkpoint_dir']
    monitor_env = gym.make(flags["game"])
    monitor_env.monitor.start(flags["eval_dir"]+"/"+flags["experiment"]+"/eval")

    # Wrap env with AtariEnvironment helper class
    env = Atari_Environment(monitor_env, flags["scaled_width"],flags["scaled_height"], flags["agent_history_length"])

    for i_episode in xrange(flags['num_eval_episodes']):
        current_state = env.reset()
        episode_return = 0
        done = False
        while not done:
            monitor_env.render()
            q_values = graph_ops["q_values"].eval(session = session, feed_dict = {graph_ops["state"] : [current_state]})
            action_index = np.argmax(q_values)
            next_state, reward, done, info = env.step(action_index)
            current_state = next_state
            episode_return += reward
        print episode_return
    monitor_env.monitor.close()


def env_state_action_space():
    env = gym.make(flags["game"])
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n
    return input_shape, num_actions

if __name__ == "__main__":
    g = tf.Graph()
    with g.as_default(), tf.Session() as session:
        obs_space, num_actions = env_state_action_space()
        graph_ops = initialize_graph_ops(num_actions)
        saver = tf.train.Saver()

        K.set_session(session)

        if flags["testing"]:
            evaluation(session, num_actions, graph_ops, saver)
        else:
            train(session, num_actions, graph_ops, saver)







