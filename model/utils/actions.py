import torch
import numpy as np


LEFT = [-1.0, 0.0, 0.0]
RIGHT = [1.0, 0.0, 0.0]
GAS = [0.0, 1.0, 0.0]
BRAKE = [0.0, 0.0, 1.0]

ACTIONS = [LEFT, RIGHT, GAS, BRAKE]


def get_action_space():
    return len(ACTIONS)


def get_actions(probs):
    values, indices = probs.max(1)
    actions = np.zeros((probs.size(0), 3))
    for i in range(probs.size(0)):
        action = ACTIONS[indices[i]]
        actions[i] = float(values[i]) * np.array(action)
    return actions


def compute_action_logs_and_entropies(probs, log_probs):
    values, indices = probs.max(1)
    indices = indices.view(-1, 1)
    action_log_probs = log_probs.gather(1, indices)
    entropies = -(log_probs * probs).sum(-1)
    return action_log_probs, entropies
