import multiprocessing
import gymnasium as gym
import torch
from multiprocessing import Process, Pipe
from .environment_wrapper import EnvironmentWrapper
import random
import numpy as np

def worker(connection, stack_size, id):
    base_seed = 42 + id
    random_gen = np.random.default_rng(base_seed)
    env = make_environment(stack_size, seed=int(base_seed))
    while True:
        command, data = connection.recv()
        if command == 'step':
            state, reward, done = env.step(data)
            if done:
                new_seed = random_gen.integers(0, 100000)
                state = env.reset(seed=int(new_seed))
            connection.send((state, reward, done))
        elif command == 'reset':
            new_seed = random_gen.integers(0, 100000)
            state = env.reset(seed=int(new_seed))
            connection.send(state)

def make_environment(stack_size, seed=42):
    env = gym.make('CarRacing-v2')
    # env = gym.make('CarRacing-v2',render_mode='human')
    env.action_space.seed(int(seed))
    env.observation_space.seed(int(seed))
    np.random.seed(seed)
    random.seed(seed)
    env_wrapper = EnvironmentWrapper(env, stack_size)
    return env_wrapper


class ParallelEnvironments:
    def __init__(self, stack_size, number_of_processes=multiprocessing.cpu_count()):
        self.number_of_processes = number_of_processes
        self.stack_size = stack_size

        # pairs of connections in duplex connection
        self.parents, self.childs = zip(*[Pipe() for _
                                          in range(number_of_processes)])
        
        
        self.processes = [
            Process(target=worker, args=(child, self.stack_size, id), daemon=True)
            for id, child in enumerate(self.childs)
        ]

        for process in self.processes:
            process.start()

    def step(self, actions):
        for action, parent in zip(actions, self.parents):
            parent.send(('step', action))
        results = [parent.recv() for parent in self.parents]
        states, rewards, dones = zip(*results)
        return torch.Tensor(states), torch.Tensor(rewards), torch.Tensor(dones)

    def reset(self):
        for parent in self.parents:
            parent.send(('reset', None))
        results = [parent.recv() for parent in self.parents]
        return torch.tensor(np.array(results), dtype=torch.float32)

    def get_state_shape(self):
        return (self.stack_size, 84, 84)


if __name__ == '__main__':
    env = ParallelEnvironments(5, number_of_processes=2)
    random_env = gym.make('CarRacing-v2')
    res = env.reset()
    for i in range(1000):
        ac = random_env.action_space.sample()
        actions = [ac, ac]
        results = env.step(actions)

        if torch.all(torch.eq(torch.Tensor(results[0][0][0]), torch.Tensor(results[0][1][0]))):
            print(i)
