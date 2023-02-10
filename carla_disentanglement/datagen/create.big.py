import argparse
import os
import pickle
import random

import numpy as np

from carla.env.env_rendering import EnvRenderer, get_gt_factors
from carla.train_agent import env_fnc
from contextual_gridworld.environment.colors import COLOR_TO_IDX, COLORS
from contextual_gridworld.environment.env import load_context_config
from general_utils.io_utils import check_dir

from PIL import Image
from tqdm import tqdm
import itertools

seed = 1
random.seed(seed)
np.random.seed(seed)


n_samples = 1_000_000
crop_border = 12
dir = './data2'

env = "MiniGrid-Contextual-v0"
context_config = './carla/env/context_configurations/pmlr_all.yaml'
reward = './carla/env/reward_configurations/pmlr.yaml'
tile_size = 12
env_grid_size = 6  # ?
grid_size = 4
n_objects = 4

max_n_obstacles = 2
max_n_goodies = 2
total_objects = max_n_goodies + max_n_obstacles + 2  # +2 for agent and goal
env_renderer = EnvRenderer(total_objects=total_objects, grid_size=env_grid_size,
                           tile_size=tile_size,
                           context_config=context_config)


contexts, subdivs = load_context_config(context_config)

agent_index = 0
goal_index = 1
goodie_start_index = goal_index+1
hidden_pos = (-1, -1)
obstacle_start_index = goal_index+max_n_goodies+1

images = []
labels = []
duplicates = 0
invalids = 0
gt_index = dict()

valid_colors = list(set([COLOR_TO_IDX[v] for c in contexts for v in c.values()]))
color_map = dict(map(lambda t: (t[1], t[0]), enumerate(valid_colors + [-1])))
valid_p = list(range(1, grid_size+1)) + [-1]
pos_map = dict(map(lambda t: (t[1], t[0]), enumerate(valid_p)))


def gt_to_i(gt):
    i = 0
    for index in range(total_objects*2):
        i += pos_map[gt[index]]
        i *= len(pos_map)

    for index in range(total_objects):
        i += color_map[gt[total_objects*2+index]]
        i *= len(color_map)

    return i


# genrate all with no goodies/obstecles
factors = list(itertools.product(
    itertools.product(range(1, grid_size+1), range(1, grid_size+1)),  # agen_pos
    itertools.product(range(1, grid_size+1), range(1, grid_size+1)),  # goal_pos
    [1,  5,  8, 11, 12, 13, 14, 15, 16, 17]  # limited contexts since obstacle color is not affected
))

for f in tqdm(factors):
    gt = np.full(total_objects * 3, -1)

    agent_pos, goal_pos, color = f

    gt[agent_index:: total_objects] = list(agent_pos) + [color]
    gt[goal_index:: total_objects] = list(goal_pos) + [COLOR_TO_IDX[contexts[0]['goal']]]

    img = env_renderer.render_gt(gt, agent_pos, 0)
    images.append(img[crop_border:-crop_border, crop_border:-crop_border])
    labels.append(gt)
    gt_index[gt_to_i(gt)] = True

positions = list(itertools.product(range(1, grid_size+1), range(1, grid_size+1)))
positions_w_hidden = list(itertools.product(range(1, grid_size+1), range(1, grid_size+1))) + [(-1, -1)]
hidden_pos_index = 16

pos_combinations = 90 * (4**2)**2 * (4**2 + 1) ** 4
factor_sizes = [90, 16, 16, 17, 17, 17, 17]


def num_to_f(num):
    f = []
    for fs in factor_sizes:
        f.append(num % fs)
        num //= fs

    return f


samples = random.sample(range(pos_combinations), n_samples)

for s in tqdm(samples):
    f = num_to_f(s)

    gt = np.full(total_objects * 3, -1)

    context_index, agent_pos, goal_pos, g1_pos, g2_pos, o1_pos, o2_pos = f
    objs = [g1_pos, g2_pos, o1_pos, o2_pos]
    try:
        while True:
            objs.remove(hidden_pos_index)
    except ValueError:
        pass

    if (agent_pos in objs or
            goal_pos in objs or
            len(set(objs)) < len(objs)):
        invalids += 1
        continue

    agent_pos = positions[agent_pos]
    goal_pos = positions[goal_pos]
    g1_pos = positions_w_hidden[g1_pos]
    g2_pos = positions_w_hidden[g2_pos]
    o1_pos = positions_w_hidden[o1_pos]
    o2_pos = positions_w_hidden[o2_pos]

    context = contexts[context_index]

    gt[agent_index:: total_objects] = list(agent_pos) + [COLOR_TO_IDX[context['agent']]]
    gt[goal_index:: total_objects] = list(goal_pos) + [COLOR_TO_IDX[context['goal']]]

    gt[goodie_start_index:: total_objects] = list(g1_pos) + [COLOR_TO_IDX[context['goodie']]]
    gt[goodie_start_index+1:: total_objects] = list(g2_pos) + [COLOR_TO_IDX[context['goodie']]]
    gt[obstacle_start_index:: total_objects] = list(o1_pos) + [COLOR_TO_IDX[context['obstacle']]]
    gt[obstacle_start_index+1:: total_objects] = list(o2_pos) + [COLOR_TO_IDX[context['obstacle']]]

    ival = gt_to_i(gt)
    if ival in gt_index:
        duplicates += 1
        continue

    img = env_renderer.render_gt(gt, agent_pos, 0)
    images.append(img[crop_border:-crop_border, crop_border:-crop_border])
    labels.append(gt)
    gt_index[ival] = True


print("total samples", len(labels))
print("duplicate gts", duplicates)
print("invalid gt samples", invalids)
np.savez_compressed(dir+'/MiniGrid-Contextual-v0.big.npz', images=images, labels=labels)
