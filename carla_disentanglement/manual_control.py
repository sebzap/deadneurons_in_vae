
import argparse
import gym

from carla_disentanglement.env.window import Window
from carla_disentanglement.env.wrapper import *


def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)


def reset():

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    if 'collected_obstacles' in obs:
        print(f"collected_obstacles: {obs['collected_obstacles']}")
        print(f"collected_goodies: {obs['collected_goodies']} \n------\n")

    redraw(obs)


def step(action):
    obs, reward, done, info = env.step(action)

    print('step=%s, reward=%.2f' % (env.step_count, reward))
    if 'collected_obstacles' in obs:
        print(f"collected_obstacles: {obs['collected_obstacles']}")
        print(f"collected_goodies: {obs['collected_goodies']} \n------\n")

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)



def key_handler(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return


parser = argparse.ArgumentParser()
parser.add_argument("--env", help="gym environment to load", default='MiniGrid-Context-Dynamic-Obstacles-8x8-v0')
parser.add_argument("--seed", type=int, help="random seed to generate the environment with", default=0)
parser.add_argument("--tile_size", type=int, help="size at which to render tiles", default=32)
parser.add_argument('--agent_view', default=False, help="draw what agent sees", action='store_true')
parser.add_argument('--no_objects', default=False, help="don't use objects", action='store_true')
parser.add_argument('--fully_obs', default=False, help="fully observable wrapper", action='store_true')
parser.add_argument("--reward", help="choose reward configuration", default='default.yaml')
parser.add_argument("--obj_move_prob", type=float, help="object move probability", default=.3)
parser.add_argument("--n_contexts", type=int,  help="number of contexts", default=5)
args = parser.parse_args()

env = gym.make(args.env, reward_config=args.reward, obj_move_prob=args.obj_move_prob,
               n_contexts=args.n_contexts, seed=args.seed)

if args.no_objects:
    env.n_goodies = 0
    env.n_obstacles = 0

if args.agent_view:
    env = RGBImgObsWrapper(env) if args.fully_obs else RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

else:
    if args.fully_obs:
        env = FullyObsWrapper(env)


window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
