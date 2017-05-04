# This python wrapper is required to pre-process the Atari Gym Environment
# It takes care of the following requirements (as specified in Mnih et al 2015 - https://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf)
#    1) The agent sees and selects actions on every 'k'th frame instead of every frame, and its last action is repeated on skipped frames.
#    2) Luminance features resized to required width and height
#    3) Return state as a history of most receent frames (m=4 in this case)

import cv2
import numpy as np
from collections import deque

class Atari_Environment():
    def __init__(self, env, scaled_width = 84, scaled_height = 84, last_m_frames = 4):
        self.env = env
        self.scaled_width = scaled_width
        self.scaled_height = scaled_height
        self.last_m_frames = last_m_frames

        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n

        self.agent_history = deque()

    def reset(self):
        # Resets the environment and clears agent history
        self.agent_history = deque()

        observation = self.env.reset()
        initial_state = np.stack([self.process_image(observation) for _ in range(self.last_m_frames)], axis=0)

        for i in range(self.last_m_frames):
            self.agent_history.append(initial_state[i,:,:])

        return initial_state

    def process_image(self, observation):
        # Converts the RGB image to grayscale and resizes it

        return cv2.resize(cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY), (self.scaled_width,self.scaled_height))
        #return cv2.resize(cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY), (0,0), fx=0.5, fy=0.5)  # If you want to resize by 50%

    def step(self, action):
        # Executes actions just like in a gym environment
        # Here we pop old frame and add new frames to the queue and return the processed state
        next_observation, reward, done, info = self.env.step(action)
        next_frame = self.process_image(next_observation)

        self.agent_history.popleft()
        self.agent_history.append(next_frame)

        next_state = np.array(self.agent_history)
        return next_state, reward, done, info

