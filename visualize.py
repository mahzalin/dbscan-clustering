import matplotlib.pyplot as plt


def visualize_dataset(dataset, labels):
    """
    Visualize the clustered dataset.

    :param dataset: np.ndarray - the dataset to visualize
    :param labels: list or np.ndarray - cluster labels assigned to each point
    """
    unique_labels = set(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for label in unique_labels:
        color = 'k' if label == -1 else colors(label)
        class_member_mask = (labels == label)
        xy = dataset[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color,
                 markeredgecolor='k', markersize=6, label=f"Cluster {label}" if label != -1 else "Noise")

    plt.title("DBSCAN Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
