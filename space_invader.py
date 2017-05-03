from pylab import *
import numpy as np
import random
import math
import gym
env = gym.make('SpaceInvaders-v0')

for i_episode in range(20):
    observation = env.reset()
    print env.observation_space.shape, env.action_space.n
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print (reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
