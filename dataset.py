import numpy as np
from sklearn.datasets import make_blobs


def generate_dataset(num_points, centers, std_dev=0.4, save_path="./clusters.npy"):
    """
    Generates a synthetic dataset using Gaussian blobs.
    :param num_points: int - total number of points to generate
    :param centers: list of tuples - coordinates of the centers of the blobs
    :param std_dev: float - standard deviation of the blobs
    :param save_path: str - file path to save the generated dataset
    :return: np.ndarray - the generated dataset
    """
    dataset, _ = make_blobs(
        n_samples=num_points,
        centers=centers,
        cluster_std=std_dev,
        center_box=(0, 1),
        random_state=42
    )
    np.save(save_path, dataset)
    return dataset
