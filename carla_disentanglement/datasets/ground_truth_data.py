from torch.utils.data import Dataset
"""
    Based on 
    https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/data/ground_truth/ground_truth_data.py#L22

    Adapted for pytorch dataset
    """


class GroundTruthDataset(Dataset):
    """Abstract class for data sets that are two-step generative models."""

    @property
    def num_factors(self):
        raise NotImplementedError()

    @property
    def factors_num_values(self):
        raise NotImplementedError()

    @property
    def observation_shape(self):
        raise NotImplementedError()

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        raise NotImplementedError()

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        raise NotImplementedError()

    def sample(self, num, random_state):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(factors, random_state)

    def sample_observations(self, num, random_state):
        """Sample a batch of observations X."""
        return self.sample(num, random_state)[1]
