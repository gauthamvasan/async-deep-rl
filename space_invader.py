from pylab import *
import numpy as np
import random
import math
import gym
import cv2
from atari_wrapper import Atari_Environment

env = gym.make('SpaceInvaders-v0')
goose = Atari_Environment(env)

for i_episode in range(2):
    observation = goose.reset()
    print observation.shape
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = goose.step(action)
        #print (reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

