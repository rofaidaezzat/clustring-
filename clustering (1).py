import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import clear_output


full_reads = pd.read_csv("players_fifa22.csv")

criteria =['Overall', 'Potential', 'WageEUR', 'Age', 'ValueEUR']

full_reads = full_reads.dropna(subset=criteria)

data = full_reads[criteria].copy()


# Scaling the data from 1 to 10
data = (data - data.min()) / (data.max() - data.min()) * 9 + 1


# Initialize random centroids
def random_centroid(data, k):
    centroids = []
    for i in range(k):
        centroid = get_random_point(data)
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)


# Get a random point from the data
def get_random_point(data):
    random_point = data.apply(get_random_value)
    return pd.Series(random_point)


# Get a random value from a column
def get_random_value(column):
    return float(np.random.choice(column))


# Label data points
def get_labels(data, centroids):
    distances = calculate_distances(data, centroids)
    return distances.idxmin(axis=1)


# Calculate distances between data points and centroids
def calculate_distances(data, centroids):
    distances = pd.DataFrame()
    for centroid_index, centroid in centroids.items():
        distance = calculate_distance(data, centroid)
        distances[centroid_index] = distance
    return distances


# Calculate Euclidean distance between a data point and a centroid
def calculate_distance(data, centroid):
    return np.sqrt(((data - centroid) ** 2).sum(axis=1))


# Update centroids
def new_centroids(data, labels, k):
    grouped_data = data.groupby(labels)
    centroids = grouped_data.mean().T
    return centroids


# Plot clusters
def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels)
    plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1])
    plt.draw()
    plt.pause(0.5)
    


plt.ion()  # Interactive mode: so we don't have to manually plot every iteration

max_iterations = 100
k = 3

# Measure start time
start_time = time.time()

centroids = random_centroid(data, k)
old_centroids = pd.DataFrame()
iteration = 1

# Main loop
while iteration < max_iterations and not np.array_equal(centroids, old_centroids):
    old_centroids = centroids.copy()

    labels = get_labels(data, centroids)
    centroids = new_centroids(data, labels, k)
    plot_clusters(data, labels, centroids, iteration)
    iteration += 1

# Measure end time
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time

print("Total execution time:", execution_time, "seconds")

plt.savefig('books_read.png')
