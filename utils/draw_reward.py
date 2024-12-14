import os

import argparse
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def smooth_curve(values, weight=0.8):
    smoothed_values = []
    last = values[0]
    for v in values:
        smoothed = last * weight + (1 - weight) * v
        smoothed_values.append(smoothed)
        last = smoothed
    return smoothed_values

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='[DRL] figure drawer')
    parser.add_argument('--exp',default=None,help='path of the experiment folder', type=str)
    args = parser.parse_args()

    log_dir = os.path.abspath(args.exp)
    model_name = os.path.basename(log_dir[:-1]).split('_24')[0]

    event_file = [f for f in os.listdir(log_dir) if f.startswith("events")][0]
    event_path = os.path.join(log_dir, event_file)

    event_acc = EventAccumulator(event_path)
    event_acc.Reload() # load event data

    scalar_tags = event_acc.Tags()["scalars"] # get the label of scalars
    data = {tag: {"steps": [], "values": []} for tag in scalar_tags} # init data dict

    # read the scalars' value
    for tag in scalar_tags:
        scalar_values = event_acc.Scalars(tag)
        for scalar in scalar_values:
            data[tag]["steps"].append(scalar.step)
            data[tag]["values"].append(scalar.value)


    steps = np.array(data['rewards']["steps"]) / 1e5
    raw_values = np.array(data['rewards']["values"])
    smoothed_values = smooth_curve(raw_values, weight=0.9)

    plt.figure(figsize=(12, 6))
    plt.plot(steps, raw_values)
    plt.title(model_name)
    plt.xlabel("Steps (1e5)")
    plt.ylabel("Q Value")
    plt.grid()
    save_path = os.path.join(log_dir,'Q_value.png')
    plt.savefig(save_path, dpi=300)
    plt.show()

    