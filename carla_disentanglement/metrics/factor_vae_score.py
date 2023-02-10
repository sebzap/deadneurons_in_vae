import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/evaluation/metrics/factor_vae.py


def factor_vae_score(vae, ds: torch.utils.data.Dataset):
    random_state = np.random

    global_variances = _compute_variances(vae, ds)

    active_dims = _prune_dims(global_variances)

    scores_dict = {}
    if not active_dims.any():
        scores_dict["train_accuracy"] = 0.
        scores_dict["eval_accuracy"] = 0.
        scores_dict["num_active_dims"] = 0
        return scores_dict

    training_votes = _generate_training_batch(ds, vae, 200,
                                              100, random_state,
                                              global_variances, active_dims)

    classifier = np.argmax(training_votes, axis=0)
    other_index = np.arange(training_votes.shape[1])
    train_accuracy = np.sum(training_votes[classifier, other_index]) * 1. / np.sum(training_votes)

    eval_votes = _generate_training_batch(ds,
                                          vae, 800,
                                          100, random_state,
                                          global_variances, active_dims)

    eval_accuracy = np.sum(eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)

    scores_dict["train_accuracy"] = train_accuracy
    scores_dict["eval_accuracy"] = eval_accuracy
    scores_dict["num_active_dims"] = len(active_dims)
    return scores_dict


def _compute_variances(vae, ds, batch_size=10000):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for (images, _) in dl:
        x = vae.encode(images).detach().cpu().numpy()
        break

    assert x.shape[0] == batch_size
    return np.var(x, axis=0, ddof=1)


def _prune_dims(variances, threshold=0.05):
    """Mask for dimensions collapsed to the prior."""
    scale_z = np.sqrt(variances)
    return scale_z >= threshold


def _generate_training_batch(ground_truth_data, representation_function,
                             batch_size, num_points, random_state,
                             global_variances, active_dims):
    """Sample a set of training samples based on a batch of ground-truth data.
    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      batch_size: Number of points to be used to compute the training_sample.
      num_points: Number of points to be sampled for training set.
      random_state: Numpy random state used for randomness.
      global_variances: Numpy vector with variances for all dimensions of
        representation.
      active_dims: Indexes of active dimensions.
    Returns:
      (num_factors, dim_representation)-sized numpy array with votes.
    """
    votes = np.zeros((ground_truth_data.num_factors, global_variances.shape[0]),
                     dtype=np.int64)
    for _ in range(num_points):
        factor_index, argmin = _generate_training_sample(ground_truth_data,
                                                         representation_function,
                                                         batch_size, random_state,
                                                         global_variances,
                                                         active_dims)
        votes[factor_index, argmin] += 1
    return votes


def _generate_training_sample(ground_truth_data, vae,
                              batch_size, random_state, global_variances,
                              active_dims):
    """Sample a single training sample based on a mini-batch of ground-truth data.
    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      vae: Function that takes observation as input and
        outputs a representation.
      batch_size: Number of points to be used to compute the training_sample.
      random_state: Numpy random state used for randomness.
      global_variances: Numpy vector with variances for all dimensions of
        representation.
      active_dims: Indexes of active dimensions.
    Returns:
      factor_index: Index of factor coordinate to be used.
      argmin: Index of representation coordinate with the least variance.
    """
    # Select random coordinate to keep fixed.
    factor_index = random_state.randint(ground_truth_data.num_factors)
    # Sample two mini batches of latent variables.
    factors = ground_truth_data.sample_factors(batch_size, random_state)
    # Fix the selected factor across mini-batch.
    factors[:, factor_index] = factors[0, factor_index]
    # Obtain the observations.
    observations = ground_truth_data.sample_observations_from_factors(
        factors, random_state)
    representations = vae.encode(observations).detach().cpu().numpy()
    local_variances = np.var(representations, axis=0, ddof=1)
    argmin = np.argmin(local_variances[active_dims] /
                       global_variances[active_dims])
    return factor_index, argmin
