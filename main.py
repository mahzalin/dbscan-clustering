from dataset import generate_dataset
from visualize import visualize_dataset
from sklearn.cluster import DBSCAN
import numpy as np


def main():
    points_number = 300
    centers = [(3, 3), (7, 7), (10, 11), (1, 5), (4, 10), (9, 5)]
    epsilon = 1.0
    min_samples = 4

    data = generate_dataset(points_number, centers)

    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan.fit(data)
    labels = dbscan.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noises = list(labels).count(-1)

    print(f"Number of clusters: {num_clusters}")
    print(f"Number of noise points: {num_noises}")

    visualize_dataset(data, labels)


if __name__ == "__main__":
    main()
