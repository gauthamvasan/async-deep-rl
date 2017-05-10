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




## Resources
I highly recommend going through atleast a few of these links. They're highly useful and serve as general background material for Deep Reinforcement Learning.

### Keras + Tensorflow 
* [Using Keras as a part of a Tensorflow Workflow](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html)
* To understand model definition - [Keras functional API Guide](https://keras.io/getting-started/functional-api-guide/)

### Conference papers, Journal articles, etc.
* [V Mnih et al. 2016 - Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
* [V Mnih et al. 2015 - Human-level control through deep reinforcement learning](https://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf)
* Optional, but one of the biggest successes of DQN -  [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

### Blog posts, Video lectures
* [Simple Reinforcement Learning with Tensorflow Part 8: Asynchronous Actor-Critic Agents (A3C)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
* [David Silver's "Deep Reinforcement Learning" lecture (Video)](http://videolectures.net/rldm2015_silver_reinforcement_learning/)
* [Nervana's Demystifying Deep Reinforcement Learning blog post](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)

## Acknowledgements
Thanks to [@coreylynch](https://github.com/coreylynch/async-rl) for his repo on async-rl. It helped me troubleshoot some of my own code when I was struggling with bugs. Plus, I started looking at Keras layers on the tensorflow workflow only after looking at his code. It definitely made my life easier!
