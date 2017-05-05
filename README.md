# async-deep-rl
This is a Tensorflow implementation of the asyncronous methods for Reinforcement Learning as described in ["Asynchronous Methods for Deep Reinforcement Learning"](http://arxiv.org/pdf/1602.01783v1.pdf). 

Disclaimer:This repo has no affiliation with Google Deepmind or the authors; it was just a simple project I was using to learn TensorFlow. Feedback is highly appreciated.


I'm running my initial experiments on the OpenAI Gym Atari Environment Space-Invader.

## Requirements
* [tensorflow](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html)
* [gym](https://github.com/openai/gym#installation)
* [gym's atari environment] (https://github.com/openai/gym#atari)
* [Keras](https://keras.io/)

## Resources
I highly recommend going through atleast a few of these links. I found these super useful as general background materials for deep Reinforcement Learning (RL):

### Keras + Tensorflow 
* [Using Keras as a part of a Tensorflow Workflow](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html)
* [Keras functional API Guide](https://keras.io/getting-started/functional-api-guide/)

### Conference papers, Journal articles, etc.
* [V Mnih et al. 2015 - Human-level control through deep reinforcement learning](https://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf)
* Optional but one of the biggest successes of DQN [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

### Blog posts, video lectures
* [Simple Reinforcement Learning with Tensorflow Part 8: Asynchronous Actor-Critic Agents (A3C)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
* [David Silver's "Deep Reinforcement Learning" lecture (Video)](http://videolectures.net/rldm2015_silver_reinforcement_learning/)
* [Nervana's Demystifying Deep Reinforcement Learning blog post](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)

## Notes/Comments


## Acknowledgements
Thanks to [@coreylynch](https://github.com/coreylynch/async-rl) for his repo on async-rl. It helped me troubleshoot some of my own code when I was struggling with bugs. Plus, I started looking at Keras layers on the tensorflow workflow only after looking at his code. It definitely made my life easier!
