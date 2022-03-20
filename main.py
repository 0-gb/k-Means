import numpy as np
import random


def assign_points(points, centres):
    assigns = []
    clustering_cost = 0
    for point in points:
        point_assignment = -1
        min_dist = 1e18
        for ind_centre, centre in enumerate(centres):
            diff = np.subtract(point, centre)
            this_dist = np.linalg.norm(diff)
            if this_dist < min_dist:
                min_dist = this_dist
                point_assignment = ind_centre
        assigns.append(point_assignment)
        clustering_cost += min_dist
    return clustering_cost, assigns


def assign_centres(data_points_locations, assignments, n_clusters):
    cluster_2_point_locations = {}
    for point_index in range(len(data_points_locations)):
        if assignments[point_index] not in cluster_2_point_locations:
            cluster_2_point_locations[assignments[point_index]] = [data_points_locations[point_index]]
        else:
            cluster_2_point_locations[assignments[point_index]].append(data_points_locations[point_index])
    return [np.mean(cluster_2_point_locations[i], 0) for i in range(n_clusters)]


def k_means(data_points, n_clusters, n_iter=1):
    """
    Performs the k-Means algorithm clustering over the provided points. The number of dimension is arbitrary - if the
    points are two dimensional then the clustering is two dimensional and if the points are three dimenstional then the
    clustering is three dimensional, etc.

    data_points: Locations of data points in their own multidimensional space.
    n_clusters: Number of clusters.
    n_iter: Number of iterations - k Means repetitions
    returns the assignments of points to clusters in a List of Integers
    """
    assert (n_iter > 0), "Iteration count must be larger than zero."
    assert (n_clusters > 0), "Cluster count must be larger than zero."
    assert (len(data_points) > n_clusters), "Too many clusters demanded by user for the provided dataset."
    best_cost = 1e18
    best_assignments = []
    for _ in range(n_iter):
        old_cost = 1e18

        clusters_centres = random.sample(data_points, n_clusters)  # randomly assign clusters
        while len(clusters_centres) != len(set([tuple(element) for element in clusters_centres])):
            clusters_centres = random.sample(data_points, n_clusters)  # if there is repetition among clusters, retry

        cost, assignments = assign_points(data_points, clusters_centres)

        while cost < old_cost:
            old_cost = cost
            clusters_centres = assign_centres(data_points, assignments, n_clusters)
            cost, assignments = assign_points(data_points, clusters_centres)

        if cost < best_cost:
            best_cost = cost
            best_assignments = assignments
    return best_assignments


if __name__ == '__main__':
    # Example of calling k_means
    random.seed(10)
    x1 = [random.sample(range(0, 50), 3) for i in range(100)]
    x2 = [random.sample(range(50, 100), 3) for i in range(100)]
    x3 = [random.sample(range(100, 150), 3) for i in range(100)]
    list_of_points = x1 + x2 + x3
    print(k_means(data_points=list_of_points, n_clusters=7, n_iter=10))
