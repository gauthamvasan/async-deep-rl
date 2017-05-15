import unittest
from atari_wrapper import  Atari_Environment
import gym

def fun(x):
    return x + 1

class MyTest(unittest.TestCase):
    def test(self):
        self.assertEqual(fun(3), 4)

class Atari_Tests(unittest.TestCase):
    def test(self):
        env = Atari_Environment(gym.make('SpaceInvaders-v0'), scaled_width=84, scaled_height = 84, last_m_frames=4)
        start_state = env.reset()
        self.assertEqual(start_state.shape, (4,84,84))
