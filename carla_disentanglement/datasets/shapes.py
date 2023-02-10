from torch.utils.data.dataloader import DataLoader
from carla_disentanglement.datasets.ground_truth_data import GroundTruthDataset
import numpy as np
from numpy.core.numeric import indices
import torch
import h5py
from disentanglement_lib.data.ground_truth.util import SplitDiscreteStateSpace


class Shapes3DDataset(GroundTruthDataset):
    """
    https://github.com/deepmind/3d-shapes

    Download from https://console.cloud.google.com/storage/browser/3d-shapes

    dataset is labeld whith true shape class

    some implementations copied from https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/shapes3d.py

    The data set was originally introduced in "Disentangling by Factorising".
    The ground-truth factors of variation are:
    0 - floor color (10 different values)
    1 - wall color (10 different values)
    2 - object color (10 different values)
    3 - object size (8 different values)
    4 - object type (4 different values)
    5 - azimuth (15 different values)
    """

    def __init__(self):
        data = h5py.File('./carla_disentanglement/data/3dshapes/3dshapes.h5', 'r')
        self.data = torch.from_numpy(data['images'][:]).permute(0, 3, 1, 2).requires_grad_(False)
        self.labels = torch.from_numpy(data['labels'][:]).requires_grad_(False)
        self._FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                                  'orientation']

        self.factor_sizes = [10, 10, 10, 8, 4, 15]
        self.latent_factor_indices = list(range(6))
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes,
                                                   self.latent_factor_indices)
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
            self.factor_sizes)

    def __getitem__(self, index):
        return self.data[index].div(255.), self.labels[index]

    def __len__(self):
        return self.data.shape[0]

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return [3, 64, 64]

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)

        ds = torch.utils.data.Subset(self, indices)
        count = len(indices)
        data_loader = DataLoader(ds, batch_size=count)
        return next(iter(data_loader))[0]

        # return self.data[indices].div(255.)


# class Shapes3DDataset(Dataset):
#     """
#     https://github.com/deepmind/3d-shapes

#     Download from https://console.cloud.google.com/storage/browser/3d-shapes

#     dataset is labeld whith true shape class
#     """

#     def __init__(self):
#         data = h5py.File('./data/3dshapes/3dshapes.h5', 'r')
#         self.data = data['images']
#         self.labels = data['labels']
#         self._FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
#                                   'orientation']

#     def __getitem__(self, index):
#         return torch.from_numpy(
#             self.data[index]/255.).transpose(-1, -2).transpose(-2, -3).float(), torch.from_numpy(
#             self.labels[index, 4:5])  # index 4: only return shape label

#     def __len__(self):
#         return self.data.shape[0]
