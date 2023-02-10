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


n_samples = 50000
crop_border = 8
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

with tqdm(total=n_samples) as pbar:
    while len(labels) < n_samples:
        gt = np.full(total_objects * 3, -1)

        n_goodies = random.randint(0, max_n_goodies)
        n_obstacles = random.randint(0, max_n_obstacles)

        context = contexts[random.randint(0, len(contexts) - 1)]

        open_pos = [x for x in itertools.product(range(1, grid_size+1), range(1, grid_size+1))]
        closed_pos = []

        agent_pos = (random.randint(1, grid_size), random.randint(1, grid_size))
        goal_pos = (random.randint(1, grid_size), random.randint(1, grid_size))
        open_pos.remove(agent_pos)
        closed_pos.append(agent_pos)
        if (goal_pos != agent_pos):
            open_pos.remove(goal_pos)
            closed_pos.append(goal_pos)

        gt[agent_index:: total_objects] = list(agent_pos) + [COLOR_TO_IDX[context['agent']]]
        gt[goal_index:: total_objects] = list(goal_pos) + [COLOR_TO_IDX[context['goal']]]

        if n_goodies == 1:
            i = random.randint(0, 1)
            pos = random.choice(open_pos)
            open_pos.remove(pos)
            closed_pos.append(pos)
            gt[goodie_start_index+i:: total_objects] = list(pos) + [COLOR_TO_IDX[context['goodie']]]
        else:
            for i in range(n_goodies):
                pos = random.choice(open_pos)
                open_pos.remove(pos)
                closed_pos.append(pos)
                gt[goodie_start_index+i:: total_objects] = list(pos) + [COLOR_TO_IDX[context['goodie']]]

        if n_obstacles == 1:
            i = random.randint(0, 1)
            pos = random.choice(open_pos)
            open_pos.remove(pos)
            closed_pos.append(pos)
            gt[obstacle_start_index+i:: total_objects] = list(pos) + [COLOR_TO_IDX[context['obstacle']]]
        else:
            for i in range(n_obstacles):
                pos = random.choice(open_pos)
                open_pos.remove(pos)
                closed_pos.append(pos)
                gt[obstacle_start_index+i:: total_objects] = list(pos) + [COLOR_TO_IDX[context['obstacle']]]

        if any(np.array_equal(gt, x) for x in labels):
            duplicates += 1
            continue

        img = env_renderer.render_gt(gt, agent_pos, 0)
        images.append(img[crop_border:-crop_border, crop_border:-crop_border])
        labels.append(gt)
        pbar.update(1)

np.savez_compressed(dir+'/MiniGrid-Contextual-v0.npz', images=images, labels=labels)
