import gym

import matplotlib.pyplot as plt

from gym.wrappers.filter_observation import FilterObservation
from carla_disentanglement.env.wrapper import *


def env_fnc(seed, env_id, rank=0, fully_obs=False, no_objects=False, contextual=False, n_contexts=1, max_ep_length=100,
            random_start=False, norm_obs=False, reward='default.yaml', context_encoding='one_hot', obj_move_prob=0.3, **kwargs):

    def build_env():
        # we can directly pass some parameters to the environment here
        env = gym.make(env_id, n_contexts=n_contexts, max_steps=max_ep_length,
                       agent_start_pos=None if random_start else (1, 1), seed=seed + rank, reward_config=reward,
                       context_enc_type=context_encoding, obj_move_prob=obj_move_prob, **kwargs)

        # important to set all seeds here
        # (note that you will also have to set your seeds in your training script e.g. for pytorch)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)

        if no_objects:
            env.n_goodies = 0
            env.n_obstacles = 0

        # filter the parts from the observation space we want to show to the agent
        # this has to be at least the image and we also decided to tell the agent the number of collected goodies
        # and obstacles as we are not yet working in a recurrent setup
        filter_keys = ['image', 'collected_goodies', 'collected_obstacles']

        # if the contextual flag is set we also tell the agent in which the environment is currently in
        if contextual:
            filter_keys.append('context')

        env = FilterObservation(env, filter_keys)

        # either show the full environment or only a partial view
        if fully_obs:
            env = RGBImgObsWrapper(env)
        else:
            env = RGBImgPartialObsWrapper(env)

        # normalize the observations
        if norm_obs:
            normalization = {'image': 255}  # scale rgb image with max value of 255 to range (0, 1)

            if context_encoding == 'numeric':
                # in case we choose a numeric context encoding the context id will be divided by the number of contexts
                # to make it to range (0, 1)
                normalization['context'] = n_contexts

            # scale number of collected objects to range (0, 1)
            normalization['collected_goodies'] = env.n_goodies
            normalization['collected_obstacles'] = env.n_obstacles

            env = NormalizeObservationByKey(env, normalization)

        return env

    return build_env


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MiniGrid-Context-Dynamic-Obstacles-8x8-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--context_encoding', choices=['one_hot', 'numeric', 'binary'], default='one_hot')
    parser.add_argument('--fully_obs', default=False, action='store_true')
    parser.add_argument('--random_start', default=False, action='store_true')
    parser.add_argument('--contextual', default=False, action='store_true')
    parser.add_argument('--no_objects', default=False, action='store_true')
    parser.add_argument('--norm_obs', default=False, action='store_true')
    parser.add_argument('--n_contexts', type=int, default=1)
    parser.add_argument('--max_ep_length', type=int, default=100)
    parser.add_argument('--reward', help='choose reward configuration', default='default.yaml')
    parser.add_argument('--obj_move_prob', type=float, default=0.3)

    args = parser.parse_args()

    # Instantiate n_proc environments (rank is provided in case you want to create multiple instances e.g. for PPO)
    env = env_fnc(env_id=args.env, seed=args.seed, rank=0, fully_obs=args.fully_obs, no_objects=args.no_objects,
                  contextual=args.contextual, n_contexts=args.n_contexts, max_ep_length=args.max_ep_length,
                  random_start=args.random_start, norm_obs=args.norm_obs, reward=args.reward,
                  context_encoding=args.context_encoding, obj_move_prob=args.obj_move_prob)()

    obs = env.reset()

    print('Observation Keys:', obs.keys())

    plt.figure()
    plt.imshow(obs['image'])
    plt.show()
