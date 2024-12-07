import numpy as np


LEFT = [-1.0, 0.0, 0.0]
RIGHT = [1.0, 0.0, 0.0]
GAS = [0.0, 1.0, 0.0]
BRAKE = [0.0, 0.0, 1.0]

ACTIONS = [LEFT, RIGHT, GAS, BRAKE]


def get_action_space():
    return len(ACTIONS)


# def get_actions(probs):
#     values, indices = probs.max(1)
#     actions = np.zeros((probs.size(0), 3))
#     for i in range(probs.size(0)):
#         action = ACTIONS[indices[i]]
#         actions[i] = float(values[i]) * np.array(action)
#     return actions

def get_actions(probs):
    direction = probs[:,:2]
    gas_brake = probs[:,2:]
    values_1, indices_1 = direction.max(1)
    values_2, indices_2 = gas_brake.max(1)
    actions = np.zeros((probs.size(0), 3))
    for i in range(probs.size(0)):
        actions[i] = float(values_1[i]) * np.array(ACTIONS[indices_1[i]]) + \
                     float(values_2[i]) * np.array(ACTIONS[indices_2[i]+2])
    return actions
