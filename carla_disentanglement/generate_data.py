from env_setup_example import env_fnc
import json
import os.path
from os import path
from PIL import Image

# python generate_data.py --seed=1337 --random_start --contextual --samples=10 --n_contexts=5 --context_encoding=numeric
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MiniGrid-Context-Dynamic-Obstacles-8x8-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--context_encoding', choices=['one_hot', 'numeric', 'binary'], default='one_hot')
    parser.add_argument('--folders', default=False, action='store_true')
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
                  random_start=args.random_start, norm_obs=args.norm_obs, reward=args.reward, grid_size=args.grid_size,
                  context_encoding=args.context_encoding, obj_move_prob=args.obj_move_prob)()

    if args.folders:
        base_path = "data/splits/"

        for i in range(args.samples):
            obs = env.reset()
            folder = base_path+str(int(obs['context'].tolist()[0]))+'/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            im = Image.fromarray(obs['image'])
            im.save(folder + str(i) + ".png")

    else:
        base_path = "data/grid/"

        labels = []
        labels_filename = base_path+"labels.json"

        if path.exists(labels_filename):
            with open(labels_filename, "r") as fp:
                labels = json.load(fp)

        for i in range(args.samples):
            obs = env.reset()
            labels.append(obs['context'].tolist())
            im = Image.fromarray(obs['image'])
            im.save(base_path + str(len(labels)-1) + ".png")

        with open(labels_filename, "w") as fp:
            json.dump(labels, fp)
