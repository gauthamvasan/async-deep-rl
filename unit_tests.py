import unittest
from atari_wrapper import  Atari_Environment
import gym

class Atari_Tests(unittest.TestCase):
    def test(self):
        env = Atari_Environment(gym.make('SpaceInvaders-v0'), scaled_width=84, scaled_height = 84, last_m_frames=4)
        start_state = env.reset()
        self.assertEqual(start_state.shape, (4,84,84))

        state, reward, done, info = env.step(1)
        self.assertEqual(state.shape, (4, 84, 84))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

class Model_Tests(unittest.TestCase):
    def test(self):
        raise NotImplementedError