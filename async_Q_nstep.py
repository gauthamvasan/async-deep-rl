#!/usr/bin/python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disables debuugging logs of tensorflow
import numpy as np
import math
import gym
import cv2
from atari_wrapper import Atari_Environment
from build_model import build_network, sequential_network
from keras import backend as K
import tensorflow as tf
import threading, time

T = 0  # Global counter
flags = {}

# Choose the Gym environment you intend to run
flags['experiment'] = 'async_dqn_n_step_space_invader'  # Name of the experiment
flags['game'] = 'SpaceInvaders-v0'  # OpenAI Gym handle/name for the game

# Algorithm specific flags and hyper-parameters
flags['num_actor_threads'] = 16  # Number of concurrent actor-learner threads to use during training.
flags['T_max'] = 20000000  # Number of training frames/steps
flags['async_update_frequency'] = 32  # Frequency with which each actor learner thread does an async gradient update
flags['target_network_update_frequency'] = 40000  # Update and Reset the target network every n timesteps
flags['learning_rate'] = 1*math.pow(10,-4)  # Initial learning rate
flags['decay_rate_RMSProp'] = 0.99
flags['gamma'] = 0.99  # Discount rate for the reward
flags['num_steps_Q'] = 5    # Denoted as t_max in the paper - Basically the value of 'n' in n-step return
flags["clip_norm"] = 10.0


# Pre-processing parameters (The RGB image is pre-processed to fit computational requirements)
flags['scaled_width'] = 84  # Scale screen to this width.
flags['scaled_height'] = 84  # Scale screen to this height.
flags['agent_history_length'] = 4  # Use this number of recent screens as the environment state.

# Sample final epsilon from the value listed below
flags["final_epsilon_choices"] = [0.1, 0.01, 0.5]
flags["final_epsilon_choice_probabilities"] = [0.4, 0.3, 0.3]
flags['anneal_epsilon_timesteps'] = 4000000  # 'Number of timesteps to anneal epsilon.

# Summary writer
flags['summary_dir'] = '/tmp/summaries'  # Directory for storing tensorboard summaries
flags['checkpoint_dir'] = '/tmp/checkpoints' + "/" + flags[
    "experiment"]  # Directory for storing model checkpoints
flags['summary_interval'] = 5  # Save training summary to file every n seconds (rounded up to statistics interval)
flags['checkpoint_interval'] = 600  # Save the parameters every n seconds
flags['eval_dir'] = '/tmp/'  # Directory to store gym evaluation
flags['checkpoint_file'] = "/filename.ckpt"  # Choose which weights to load

# Testing & Rendering
flags['render_training'] = False  # If True, have gym render environments during training
flags['testing'] = False  # If True, run gym evaluation
flags['num_eval_episodes'] = 100  # Number of episodes to run gym evaluation


def initialize_graph_ops(num_actions):
    # Shared network initializations
    state = tf.placeholder("float",
                           [None, flags['agent_history_length'], flags['scaled_width'], flags['scaled_height']])
    shared_q_network = sequential_network(num_actions, flags['agent_history_length'], flags['scaled_width'], flags['scaled_height'])
    network_params = shared_q_network.trainable_weights
    q_values = shared_q_network(state)

    # Target network initializations (for async updates)
    target_state = tf.placeholder("float",
                                  [None, flags['agent_history_length'], flags['scaled_width'], flags['scaled_height']])
    target_q_network = sequential_network(num_actions, flags['agent_history_length'], flags['scaled_width'], flags['scaled_height'])
    target_network_params = target_q_network.trainable_weights
    target_q_values = target_q_network(target_state)

    # Op for async updates of target network with shared/online network weights
    async_update_target_network = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]

    # Cost and gradient update ops
    y = tf.placeholder("float", [None])                       # Target
    actions_list = tf.placeholder(tf.int32, [None, None])     # 2D list consisting of sample number (in th batch) and the action chosen
    action_values = tf.gather_nd(q_values,actions_list)       # Gather the Q values

    cost = tf.reduce_mean(tf.square(y - action_values))
    optimizer = tf.train.RMSPropOptimizer(flags["learning_rate"], decay=flags['decay_rate_RMSProp'])
    gradient_update = optimizer.minimize(cost,var_list=network_params)

    # Thread networks
    thread_networks = []
    thread_network_params = []
    thread_q_values = []
    copy_network_to_thread = []
    thread_action_values = []
    thread_costs = []
    thread_compute_gradients = []
    async_update_shared_network = []

    for i in range(flags["num_actor_threads"]):
        thread_networks.append(sequential_network(num_actions, flags['agent_history_length'], flags['scaled_width'], flags['scaled_height']) )
        thread_network_params.append(thread_networks[i].trainable_weights)
        thread_q_values.append(thread_networks[i](state))

        copy_network_to_thread.append([thread_network_params[i][j].assign(network_params[j]) for j in range(len(thread_network_params[i]))])

        thread_action_values.append(tf.gather_nd(thread_q_values[i], actions_list))
        thread_costs.append(tf.reduce_mean(tf.square(y - thread_action_values[i])))
        thread_compute_gradients.append(tf.gradients(thread_costs[i],thread_network_params[i]))

        grad_and_vars = zip(thread_compute_gradients[i], network_params)
        gradients, variables = zip(*grad_and_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, flags["clip_norm"])
        grad_and_vars = zip(gradients, variables)
        async_update_shared_network.append(optimizer.apply_gradients(grad_and_vars))



    graph_ops = {"q_values": q_values,
                 "network_params": network_params,
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

                 "thread_networks": thread_networks,
                 "thread_network_params": thread_network_params,
                 "thread_q_values": thread_q_values,
                 "copy_network_to_thread": copy_network_to_thread,
                 "async_update_shared_network": async_update_shared_network,

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
    t = 0  # Thread step counter
    summary_placeholders, update_ops, summary_op = summary_ops

    # Selecting epsilon as described in the paper
    epsilon = 1.0
    final_epsilon = np.random.choice(flags["final_epsilon_choices"], p=flags["final_epsilon_choice_probabilities"])
    epsilon_anneal_factor = (epsilon - final_epsilon) / flags['anneal_epsilon_timesteps']

    action_set = np.arange(num_actions)

    # For n-step Q learning, there is an additional set of network weights maintained for the thread
    thread_q_values = graph_ops["thread_q_values"][thread_id]
    copy_network_to_thread = graph_ops["copy_network_to_thread"][thread_id]
    async_update_shared_network = graph_ops["async_update_shared_network"][thread_id]

    while T < flags["T_max"]:
        current_state = env.reset()
        done = False

        episode_return = 0.0
        avg_episode_max_q = 0.0
        episode_timesteps = 0

        while not done:
            # Clear the gradients
            y_mem = []
            action_mem = []
            state_mem = []
            reward_mem = []

            # Initialize thread specific parameters
            session.run(copy_network_to_thread)
            t_start = t

            while t - t_start < flags["num_steps_Q"] and not done:
                q_vals = thread_q_values.eval(session=session, feed_dict={graph_ops["state"]: [current_state]})
                action = sample_action_epsilon_greedy(q_vals, action_set, epsilon)

                next_state, reward, done, info = env.step(action=action)
                scaled_reward = np.clip(reward, -1, 1)

                # Update memory
                state_mem.append(current_state)
                action_mem.append([t - t_start, action])
                reward_mem.append(scaled_reward)

                # Summary writer info updates
                episode_timesteps += 1
                episode_return += reward
                avg_episode_max_q += np.max(q_vals)

                # Update counters
                T += 1
                t += 1

                # Update the state
                current_state = next_state

                # Anneal epsilon
                if epsilon > final_epsilon:
                    epsilon -= epsilon_anneal_factor

            if done:
                R = 0

                # Print statistics of the agent's performance this episode
                stats = [episode_return + reward, (avg_episode_max_q + np.max(q_vals)) / float(episode_timesteps),
                         epsilon, episode_timesteps]
                for i in range(len(stats)):
                    session.run(update_ops[i], feed_dict={summary_placeholders[i]: float(stats[i])})

                print "THREAD:", thread_id, "/ TIME", T, "/ TIMESTEP", t, "/ EPSILON", epsilon, "/ REWARD", episode_return, "/ Q_MAX %.4f" % (
                    avg_episode_max_q / float(episode_timesteps)), "/ EPSILON PROGRESS", t / float(
                    flags["anneal_epsilon_timesteps"])
            else:
                target_Q = graph_ops["target_q_values"].eval(session=session,
                                                                   feed_dict={graph_ops["target_state"]: [current_state]})
                R = np.max(target_Q)

            for i in range(t-t_start-1,-1,-1):
                R += reward_mem[i] + flags["gamma"]*R
                y_mem.append(R)

            y_mem.reverse()

            # Async update of global network
            session.run(async_update_shared_network, feed_dict={graph_ops["y"]: y_mem,
                                                                graph_ops["state"]: state_mem,
                                                                graph_ops["action_list"]: action_mem})



            # Updates of target network
            if T % flags["target_network_update_frequency"] == 0:
                session.run(graph_ops["async_update_target_network"])

            # Save the model every 'n' seconds
            if t % flags['checkpoint_interval'] == 0:
                saver.save(session, flags["checkpoint_dir"])


def train(session, num_actions, graph_ops, saver):
    global T
    summary_placeholders, update_ops, summary_op = initialize_summary_ops()

    # Initialize tensorflow variables
    session.run(tf.initialize_all_variables())

    session.run(graph_ops["async_update_target_network"])
    envs = [Atari_Environment(gym.make('SpaceInvaders-v0'), flags["scaled_width"], flags["scaled_height"],
                              flags["agent_history_length"]) for i in range(flags["num_actor_threads"])]

    summary_save_path = flags["summary_dir"] + "/" + flags["experiment"]
    writer = tf.summary.FileWriter(summary_save_path, session.graph)
    if not os.path.exists(flags["checkpoint_dir"]):
        os.makedirs(flags["checkpoint_dir"])

    # Start num_concurrent actor-learner training threads
    actor_learner_threads = [threading.Thread(target=actor_learner, args=(
        thread_id, envs[thread_id], session, graph_ops, num_actions, [summary_placeholders, update_ops, summary_op],
        saver))
                             for thread_id in range(flags["num_actor_threads"])]
    for t in actor_learner_threads:
        t.start()

    # Show the agents training and write summary statistics
    last_summary_time = 0
    while T <= flags["T_max"]:
        if flags["render_training"]:
            print "Something failed"
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
    saver.restore(session, flags['checkpoint_dir'] + flags['checkpoint_file'])
    print "Restored model weights from ", flags['checkpoint_dir']
    monitor_env = gym.make(flags["game"])
    monitor_env.monitor.start(flags["eval_dir"] + "/" + flags["experiment"] + "/eval")

    # Wrap env with AtariEnvironment helper class
    env = Atari_Environment(monitor_env, flags["scaled_width"], flags["scaled_height"], flags["agent_history_length"])

    for i_episode in xrange(flags['num_eval_episodes']):
        current_state = env.reset()
        episode_return = 0
        done = False
        while not done:
            monitor_env.render()
            q_values = graph_ops["q_values"].eval(session=session, feed_dict={graph_ops["state"]: [current_state]})
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







