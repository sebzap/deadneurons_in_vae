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
labels_simple = []

factors = list(itertools.product(
    # itertools.product(range(1, grid_size+1), range(1, grid_size+1)),  # agen_pos
    # itertools.product(range(1, grid_size+1), range(1, grid_size+1)),  # goal_pos
    # itertools.product(range(1, grid_size+1), range(1, grid_size+1)),  # goodie_pos
    range(1, grid_size+1),
    range(1, grid_size+1),
    range(1, grid_size+1),
    range(1, grid_size+1),
    range(1, grid_size+1),
    range(1, grid_size+1),
    [1,  5,  8, 11, 12, 13, 14, 15, 16, 17],
    [1,  5,  8, 11, 12, 13, 14, 15, 16, 17]
))

context = contexts[0]
for f in tqdm(factors):
    gt = np.full(total_objects * 3, -1)

    ax, gx, ox, ay, gy, oy, color_agent, color_object = f
    agent_pos, goal_pos, g_pos = (ax, ay), (gx, gy), (oy, oy)

    gt[agent_index:: total_objects] = list(agent_pos) + [color_agent]
    gt[goal_index:: total_objects] = list(goal_pos) + [COLOR_TO_IDX[context['goal']]]
    gt[goodie_start_index:: total_objects] = list(g_pos) + [color_object]

    img = env_renderer.render_gt(gt, agent_pos, 0)
    images.append(img[crop_border:-crop_border, crop_border:-crop_border])
    labels.append([ax, gx, ox, ay, gy, oy, color_agent, color_object])

np.savez_compressed(dir+'/MiniGrid-Contextual-v0.simple.npz', images=images, labels=labels)
