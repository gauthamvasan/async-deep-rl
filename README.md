# async-deep-rl
This is a Tensorflow implementation of the asyncronous methods for Reinforcement Learning as described in ["Asynchronous Methods for Deep Reinforcement Learning"](http://arxiv.org/pdf/1602.01783v1.pdf). 

Disclaimer:This repo has no affiliation with Google Deepmind or the authors; it was just a simple project I was using to learn TensorFlow. Feedback is highly appreciated.


I'm running my initial experiments on the OpenAI Gym Atari Environment Space-Invader.

## Requirements
* [tensorflow](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html)
* [gym](https://github.com/openai/gym#installation)
* [gym's atari environment] (https://github.com/openai/gym#atari)
* Keras

## Resources
I highly recommend going through atleast a few of these links. I found these super useful as general background materials for deep Reinforcement Learning (RL):

* Optional but one of the biggest successes of DQN [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Simple Reinforcement Learning with Tensorflow Part 8: Asynchronous Actor-Critic Agents (A3C)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
* [David Silver's "Deep Reinforcement Learning" lecture](http://videolectures.net/rldm2015_silver_reinforcement_learning/)
* [Nervana's Demystifying Deep Reinforcement Learning blog post](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)

## Important notes
* In the paper the authors mention "for asynchronous methods we average over the best 5 models from **50 experiments**". I overlooked this point when I was writing this, but I think it's important. These async methods seem to vary in performance a lot from run to run (at least in my implementation of them!). I think it's a good idea to run multiple seeded versions at the same time and average over their performance to get a good picture of whether or not some architectural change is good or not. Equivalently don't get discouraged if you don't see performance on your task right away; try rerunning the same code a few more times with different seeds.

