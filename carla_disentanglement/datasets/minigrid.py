from torch.utils.data.dataloader import DataLoader
from carla_disentanglement.datasets.ground_truth_data import GroundTruthDataset
import numpy as np
from numpy.core.numeric import indices
import torch
from disentanglement_lib.data.ground_truth.util import SplitDiscreteStateSpace, StateSpaceAtomIndex


class MinigridDataset(GroundTruthDataset):
    """
    Generated without duplicates

    this is the order of variables:
     x,y,color

     and this is the order of objects:
     agent, goal, goodie1, goodie2, obstacle1, obstacle2

     in details, this is how the array is filled:
     agent_x, goal_x, ...., agent_y, goal_y,...., agent_colour, goal_colour ....


    """

    def __init__(self, pad=False):
        self.valid_colors = [1, 5, 6, 8, 11, 12, 13, 14, 15, 16, 17]
        self.n_objects = 6

        data = np.load("./carla_disentanglement/data/minigrid/MiniGrid-Contextual-v0.big.npz")
        d = data['images']
        if pad:
            d = np.pad(d, ((0, 0), (8, 8), (8, 8), (0, 0)), constant_values=100)
        self.data = torch.from_numpy(d).permute(0, 3, 1, 2).requires_grad_(False)
        labels = np.delete(data['labels'], self.n_objects*2 + 1, axis=1)  # remove goal color since it is always 6 (lime)
        self.labels = torch.from_numpy(labels).requires_grad_(False)

    def __getitem__(self, index):
        return self.data[index].div(255.), self.labels[index]

    def __len__(self):
        return self.data.shape[0]

    @property
    def num_factors(self):
        return self.n_objects * 3-1

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return [3, 48, 48]


class MinigridScoreDataset(GroundTruthDataset):
    """
    Generated with every possibility but only 0-1 goodies and no obstacels

    this is the order of variables:
     x,y,color

     and this is the order of objects:
     agent, goal, goodie1, goodie2, obstacle1, obstacle2

     in details, this is how the array is filled:
     agent_x, goal_x, ...., agent_y, goal_y,...., agent_colour, goal_colour ....


    """

    def __init__(self, pad=False):
        self.valid_colors = [1,  5,  8, 11, 12, 13, 14, 15, 16, 17]
        self.n_objects = 3

        data = np.load("./carla_disentanglement/data/minigrid/MiniGrid-Contextual-v0.simple.npz")
        d = data['images']
        if pad:
            d = np.pad(data['images'], ((0, 0), (8, 8), (8, 8), (0, 0)), constant_values=100)
        self.data = torch.from_numpy(d).permute(0, 3, 1, 2).requires_grad_(False)

        # org_n_objects = 6
        # labels = np.delete(data['labels'], [
        #     3, 4, 5,  # unused x pos
        #     org_n_objects+3, org_n_objects+4, org_n_objects+5,  # unused y pos
        #     org_n_objects*2 + 1,  # remove goal color since it is always 6 (lime)
        #     org_n_objects*2 + 2,  # remove goody color since it is always equal to agent color
        #     org_n_objects*2+3, org_n_objects*2+4, org_n_objects*2+5  # unused color
        # ], axis=1)
        # self.labels = torch.from_numpy(labels).requires_grad_(False)
        self.labels = torch.from_numpy(data['labels']).requires_grad_(False)

        self.factor_sizes = np.concatenate((
            np.full(2, 4),  # agent/goal, x positions
            np.full(self.n_objects - 2, 4),  # goodie/obstecle, x postion
            np.full(2, 4),  # agent/goal, y positions
            np.full(self.n_objects - 2, 4),  # goodie/obstecle, y postion
            np.full(1, len(self.valid_colors)),  # agent color
            np.full(1, len(self.valid_colors)),  # agent goodie/obsticle color
        ), axis=0)
        self.latent_factor_indices = list(range(self.n_objects * 3 - 1))

        self.state_space = SplitDiscreteStateSpace(self.factor_sizes,
                                                   self.latent_factor_indices)
        # print(self.factor_sizes, self.labels.shape)
        # self.index = StateSpaceAtomIndex(self.factor_sizes, self.labels.numpy())
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
            self.factor_sizes)

    def __getitem__(self, index):
        return self.data[index].div(255.), self.labels[index]

    def __len__(self):
        return self.data.shape[0]

    @property
    def num_factors(self):
        return self.n_objects * 3 - 1

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return [3, 48, 48]

    def sample(self, num, random_state):
        """Sample a batch of factors Y and observations X."""
        indices = random_state.choice(self.__len__(), num, replace=False)

        ds = torch.utils.data.Subset(self, indices)
        count = len(indices)
        data_loader = DataLoader(ds, batch_size=count)
        o, f = next(iter(data_loader))

        return f, o

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        # indices = self.index.features_to_index(all_factors)

        ds = torch.utils.data.Subset(self, indices)
        count = len(indices)
        data_loader = DataLoader(ds, batch_size=count)
        return next(iter(data_loader))[0]
