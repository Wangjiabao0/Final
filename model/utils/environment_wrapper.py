import gymnasium as gym
import numpy as np
from PIL import Image
from collections import deque
import PIL
import numpy as np

def crop(img, bottom=12, left=6, right=6):
    height, width = img.shape
    return img[0: height - bottom, left: width - right]

class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env, stack_size):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = deque([], maxlen=stack_size)

    def reset(self, seed=None):
        state, _ = self.env.reset(seed=seed)
        for _ in range(self.stack_size):
            self.frames.append(self.preprocess(state))
        return self.state()

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        preprocessed_state = self.preprocess(state)
        self.frames.append(preprocessed_state)
        return self.state(), reward, done

    def state(self):
        return np.stack(self.frames, axis=0)

    def preprocess(self, state):
        preprocessed_state = np.dot(state, [0.299, 0.587, 0.144]) - 127.0
        preprocessed_state = crop(preprocessed_state)
        return preprocessed_state

    def get_state_shape(self):
        return (self.stack_size, 84, 84)

if __name__ == '__main__':
    env = gym.make('CarRacing-v2')
    print("Action space:", env.action_space)
    print("Action space low:", env.action_space.low)
    print("Action space high:", env.action_space.high)
    env_wrapper = EnvironmentWrapper(env, 5)
    env_wrapper.reset()
    action = [0, 0, 0]
    for i in range(100):
        s, _, _ = env_wrapper.step(action)
        print(s.shape)
