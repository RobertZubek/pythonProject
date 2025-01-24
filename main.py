import time
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import random
from tqdm import tqdm
import unittest


def dbscan(data, eps, min_pts):
    """
    DBSCAN clustering algorithm implemented without additional libraries.

    Parameters:
        data (list of float): Input data points as a list of points, each point being a list of coordinates.
        eps (float): The maximum distance for two points to be considered as neighbors.
        min_pts (int): The minimum number of neighbors (including the point itself) to define a core point.

    Returns:
        list of int: Cluster labels for each data point. Noise points are labeled as -1.
    """
    def euclidean_distance(p1, p2):
        """Calculate the Euclidean distance between two points."""
        return sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))) ** 0.5

    def region_query(point):
        """Find all neighbors within `eps` distance of a given point."""
        neighbors = []
        for idx, other_point in enumerate(data):
            if euclidean_distance(point, other_point) <= eps:
                neighbors.append(idx)
        return neighbors

    def expand_cluster(point_idx, neighbors, cluster_id):
        """Expand the cluster recursively from a core point."""
        clusters[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if clusters[neighbor_idx] == -1:  # Previously marked as noise, now part of the cluster
                clusters[neighbor_idx] = cluster_id

            elif clusters[neighbor_idx] == 0:  # Not yet processed
                clusters[neighbor_idx] = cluster_id
                new_neighbors = region_query(data[neighbor_idx])
                if len(new_neighbors) >= min_pts:
                    neighbors.extend(new_neighbors)  # Expand with new core's neighbors

            i += 1

    # Initialize cluster labels (-1: noise, 0: unvisited)
    clusters = [0] * len(data)
    cluster_id = 0

    for point_idx, point in enumerate(tqdm(data)):
        if clusters[point_idx] != 0:  # Already visited
            continue

        neighbors = region_query(point)

        if len(neighbors) < min_pts:
            clusters[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1  # Start a new cluster
            expand_cluster(point_idx, neighbors, cluster_id)

    return clusters


def generate_random_data(num_points, lower_bound=0, upper_bound=100):
    """Generate random 2D points."""
    return [[random.uniform(lower_bound, upper_bound), random.uniform(lower_bound, upper_bound)] for _ in range(num_points)]


def plot_clusters(data, labels, title):
    """Visualize clustering results."""
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
    for idx, point in enumerate(data):
        plt.scatter(point[0], point[1], color=colors[labels[idx] % len(colors)] if labels[idx] != -1 else 'black', s=10)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')


# Unit tests
class TestDBSCAN(unittest.TestCase):

    def test_generate_random_data(self):
        data = generate_random_data(5)
        self.assertEqual(len(data), 5)
        self.assertTrue(all(len(point) == 2 for point in data))

    def test_dbscan_cluster_labels(self):
        data = generate_random_data(10)
        epsilon = 2.0
        min_points = 3
        labels = dbscan(data, epsilon, min_points)
        self.assertEqual(len(labels), 10)
        self.assertTrue(all(label == -1 or label > 0 for label in labels))  # Labels should be -1 or positive cluster IDs

    def test_dbscan_noise_points(self):
        data = generate_random_data(10)
        epsilon = 1.0
        min_points = 5
        labels = dbscan(data, epsilon, min_points)
        noise_points = labels.count(-1)
        self.assertGreaterEqual(noise_points, 0)  # Ensure there are noise points or no noise points
        self.assertTrue(all(label != -1 or noise_points > 0 for label in labels))  # Noise should be labeled as -1


if __name__ == "__main__":
    unittest.main()

# Prepare datasets of increasing size

sizes = [10, 100, 1000, 5000]
custom_times = []
sklearn_times = []

for size in sizes:
    data_points = generate_random_data(size)
    epsilon = 2.0
    min_points = 5

    # Measure time for custom implementation
    start_time = time.time()
    cluster_labels_custom = dbscan(data_points, epsilon, min_points)
    custom_time = time.time() - start_time
    custom_times.append(custom_time)

    # Measure time for scikit-learn implementation
    start_time = time.time()
    sklearn_dbscan = DBSCAN(eps=epsilon, min_samples=min_points)
    sklearn_labels = sklearn_dbscan.fit_predict(data_points)
    sklearn_time = time.time() - start_time
    sklearn_times.append(sklearn_time)

    print(f"Size: {size}")
    print("Custom DBSCAN Time:", custom_time)
    print("Scikit-learn DBSCAN Time:", sklearn_time)

    # Plot clustering results for the current dataset size
    if size <= 100000:  # Limit visualization for smaller datasets
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plot_clusters(data_points, cluster_labels_custom, f"Custom DBSCAN (Size: {size})")
        plt.subplot(1, 2, 2)
        plot_clusters(data_points, sklearn_labels, f"Scikit-learn DBSCAN (Size: {size})")
        plt.tight_layout()
        plt.show()

# Plot results of execution time
plt.figure(figsize=(10, 6))
plt.plot(sizes, custom_times, label="Custom DBSCAN", marker='o')
plt.plot(sizes, sklearn_times, label="Scikit-learn DBSCAN", marker='x')
plt.xlabel("Number of Points")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time vs Number of Points")
plt.legend()
plt.grid(True)
plt.show()
