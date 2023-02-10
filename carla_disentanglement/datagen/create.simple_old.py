import numpy as np

from carla.env.env_rendering import EnvRenderer
from contextual_gridworld.environment.colors import COLOR_TO_IDX
from contextual_gridworld.environment.env import load_context_config

from tqdm import tqdm
import itertools

# creates dataset with all combinations (but no obstacles and only one goodie)


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

factors = list(itertools.product(
    range(len(contexts)),
    itertools.product(range(1, grid_size+1), range(1, grid_size+1)),  # agen_pos
    itertools.product(range(1, grid_size+1), range(1, grid_size+1))  # goal_pos
))


for f in tqdm(factors):
    gt_base = np.full(total_objects * 3, -1)

    context_index, agent_pos, goal_pos = f
    context = contexts[context_index]

    gt_base[agent_index:: total_objects] = list(agent_pos) + [COLOR_TO_IDX[context['agent']]]
    gt_base[goal_index:: total_objects] = list(goal_pos) + [COLOR_TO_IDX[context['goal']]]

    open_pos = [x for x in itertools.product(range(1, grid_size+1), range(1, grid_size+1))]
    open_pos.remove(agent_pos)
    if (goal_pos != agent_pos):
        open_pos.remove(goal_pos)

    for g_pos in open_pos:
        gt = gt_base.copy()

        gt[goodie_start_index:: total_objects] = list(g_pos) + [COLOR_TO_IDX[context['goodie']]]
        # if o_pos != hidden_pos:
        #     gt[obstacle_start_index:: total_objects] = list(o_pos) + [COLOR_TO_IDX[context['obstacle']]]

        img = env_renderer.render_gt(gt, agent_pos, 0)
        images.append(img[crop_border:-crop_border, crop_border:-crop_border])
        labels.append(gt)

np.savez_compressed(dir+'/MiniGrid-Contextual-v0.simple.npz', images=images, labels=labels)
