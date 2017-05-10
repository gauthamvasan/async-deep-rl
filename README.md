# async-deep-rl
This is a Tensorflow implementation of the asyncronous methods for Reinforcement Learning as described in ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/pdf/1602.01783.pdf). 

Disclaimer:This repo has no affiliation with Google Deepmind or the authors; it was just a simple project I was using to learn TensorFlow. Feedback is highly appreciated.

## Requirements
* [tensorflow](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html)
* [OpenAI Gym](https://github.com/openai/gym#installation)
* [OpenAI Gym's Atari Environment] (https://gym.openai.com/envs#atari)
* [Keras](https://keras.io/)
* [OpenCV](http://opencv.org/)

**Note**: I'm using opencv only to convert images from RGB to Grayscale and rescale their sizes. These could easily be done in other libraries like scikit-image, etc.  

## How to use the code?

There are 2 support files:
```python
atari_wrapper.py
build_model.py
```

The **atari_wrapper.py** file is used to pre-process the screen image obtained from the game. I implemented this pre-processing module based on the description provided in [V Mnih et al. 2015](https://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf). Here's a summary of the pre-processing steps involved:

* Input frame/image dimensions = (210, 160, 3) with a 128-colour palette (basically an RGB image). 
* We convert the RGB frame to Grayscale and rescale it to (84,84) in order to make it computationally less expensive. 
* We stack the **m** most recent frames and stack them to produce the input to the Q-function, in which m = 4. (According to the paper, you can try different values of **m**)

The **build_model.py** is used to define the structure of the neural net we intend to use in this repo. I've 2 functions within which practically perform the same operation. One shows how to define a neural net explicitly in Keras. The other defines tensorflow-type layers integrated with Keras. Regardless of the function you choose to use, this is the network architecture:

* A convolutional layer with 16 filters of size (8,8) with stride 4 followed by 
* Second convolutional layer with with 32 filters of size (4,4) with stride 2 followed by 
* A fully connected layer with 256 hidden units, followed by 
* An output layer with a single linear output unit for each action representing the action-value (i.e., Q value). 
* All the hidden layers are followed by a rectifier non-linearity (ReLU activation). 

### Asynchronous 1-step Q learner

```python
python async_Q_1step.py
```

### Asynchronous n-step Q learner

```python
python async_Q_nstep.py
```

## General notes on both scripts 
This script contains all the modules relevant to training and evaluating the network. It's a headache to use multiple graphs on tensorflow. Hence, I define all the necessary graph operations in the function **initialize_graph_ops(args)**. It's saves a lot of trouble if you can track the learning progress over time. Hence, all necessary ops for the tensorboard are initialized in **initalize_summary_ops(args)**. To launch the tensorboard:

```
tensorboard --logdir /tmp/summaries/async_dqn_n_step_space_invader/
``` 
where  __async_dqn_n_step_space_invader__ is the name of the experiment. The summary and checkpoint directories, hyper-parameters and meta-data(including name of the expt) are all defined in a dictionary called **flags**. Initially I planned to use "tf.app.flags" similar to [@coreylynch's](https://github.com/coreylynch/async-rl) implementation. But there is no good documentation for it. This [stackoverflow answer](http://stackoverflow.com/questions/33932901/whats-the-purpose-of-tf-app-flags-in-tensorflow) suggests using python-gflags or argsparse instead.

The function **actor_learner(args)** is the code for each actor-learner thread as defined in the paper. The **train(args)** function initializes an atari environment for each thread and starts running the threads in parallel. I haven't implemented it with thread locking or __multi-processing__ yet. The current implementation is a bare bones version using the basic **threading** module in python. By default, flags to "render_environment" are set to False. 

For evaluation, set the "testing" flag to True. It'll load the checkpoint file (.ckpt file) specified in the "checkpoint_file" flag. 
  
### Notes & Comments on the Asynchronous n-step Q-learner 
Key featurws of the asynchronous n-step Q-learner:
* Each thread maintains its own copy of the global shared parameter vector &theta; 
* As the name suggests, it's an n-step return, which means we rollout state-action pairs until 'n' steps before we use this data to learn/update the weights. 
* At the start of the main loop, synchronize thread-specific parameters &theta;' = &theta;
* A key point to keep in mind is that we calculate gradients based on the thread-specific network &theta;'. But these gradients are applied to the global shared network &theta;.
* Usually, we can compute and apply gradients in tensorflow using the **optimizer.minimize(loss,var_list)** function. This basically computes the gradients and applies it to the relevant variable. Instead we have a workaround in this case (pseudo-code for each thread):

```python
optimizer = tf.train.RMSPropOptimizer(flags['learning_rate'], decay=flags['decay_rate_RMSProp'])

thread_network_params = thread_network.trainable_weights

grads = tf.gradients(thread_cost,thread_network_params)
clipped_grads = tf.clip_by_global_norm(grads, flags['clip_norm_value'])
vars = global_shared_network.trainable_weights

grads_and_vars = zip(clipped_grads, vars)
async_update_shared_network = optimizer.apply_gradients(grads_and_vars)
``` 

* The target for the network output is a part of the TD Error/Bellman Error - reward + maxQ. This maxQ is calculated using the global shared target parameter vector &theta;-
* This global shared target parameter vector &theta;- is updates every 40,000 frames. 
* Note: The authors of the paper mention that thet clip the gradients. But they haven't mentioned their clipping threshold. I have just used a random value here. Only a complete parameter search would elaborate its effect. 


## Resources
I highly recommend going through atleast a few of these links. They're highly useful and serve as general background material for Deep Reinforcement Learning.

### Keras + Tensorflow 
* [Using Keras as a part of a Tensorflow Workflow](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html)
* To understand model definition - [Keras functional API Guide](https://keras.io/getting-started/functional-api-guide/)

### Books, Conference papers, Journal articles, etc.
* [Reinforcement Learning: An Introduction by R.S. Sutton and A.G. Barto (Free online edition of the book)](http://incompleteideas.net/sutton/book/the-book.html)
* [V Mnih et al. 2016 - Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
* [V Mnih et al. 2015 - Human-level control through deep reinforcement learning](https://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf)
* Optional, but one of the biggest successes of DQN -  [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

### Blog posts, Video lectures
* [Simple Reinforcement Learning with Tensorflow Part 8: Asynchronous Actor-Critic Agents (A3C)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
* [David Silver's "Deep Reinforcement Learning" lecture (Video)](http://videolectures.net/rldm2015_silver_reinforcement_learning/)
* [Nervana's Demystifying Deep Reinforcement Learning blog post](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)

## Acknowledgements
Thanks to [@coreylynch](https://github.com/coreylynch/async-rl) for his repo on async-rl. It helped me troubleshoot some of my own code when I was struggling with bugs. Plus, I started looking at Keras layers on the tensorflow workflow only after looking at his code. It definitely made my life easier!
