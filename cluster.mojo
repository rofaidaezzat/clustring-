from python import Python
from python import PythonObject

fn main() raises:
    # Import necessary Python modules
    var pd = Python.import_module("pandas")
    var np = Python.import_module("numpy")
    var sklearn = Python.import_module("sklearn")
    var PCA = sklearn.decomposition.PCA
    var plt = Python.import_module("matplotlib.pyplot")
    var time = Python.import_module("time")

    # Read CSV file
    var full_reads = pd.read_csv("players_fifa22.csv")
    var criteria: PythonObject = ['Overall', 'Potential', 'WageEUR', 'Age', 'ValueEUR']
    full_reads = full_reads.dropna(subset=criteria)
    var data = full_reads[criteria].copy()

    # Scaling from 1 to 10
    data = (data - data.min()) / (data.max() - data.min()) * 9 + 1

    # Function to get a random value from a column
    def get_random_value(column: PythonObject) -> PythonObject:
        var column_list = column.tolist()
        return np.random.choice(column_list)

    # Function to get a random point from the data
    def get_random_point(data: PythonObject) -> PythonObject:
        return data.apply(get_random_value)

    # Function to initialize random centroids
    def random_centroid(data: PythonObject, k: Int) -> PythonObject:
        var centroids = []
        for i in range(k):
            var centroid = get_random_point(data)
            centroids.append(centroid)
        return pd.concat(centroids, axis=1)

    # Function to calculate Euclidean distance between a data point and a centroid
    def calculate_distance(data: PythonObject, centroid: PythonObject) -> PythonObject:
        return np.sqrt(((data - centroid) ** 2).sum(axis=1))

    # Function to calculate distances between data points and centroids
    def calculate_distances(data: PythonObject, centroids: PythonObject) -> PythonObject:
        var distances = pd.DataFrame()
        for centroid_index, centroid in centroids.items():
            var distance = calculate_distance(data, centroid)
            distances[centroid_index] = distance
        return distances

    # Function to label data points
    def get_labels(data: PythonObject, centroids: PythonObject) -> PythonObject:
        var distances = calculate_distances(data, centroids)
        return distances.idxmin(axis=1)

    # Function to update centroids
    def new_centroids(data: PythonObject, labels: PythonObject, k: Int) -> PythonObject:
        var grouped_data = data.groupby(labels)
        return grouped_data.mean().T

    # Function to plot clusters
    def plot_clusters(data: PythonObject, labels: PythonObject, centroids: PythonObject, iteration: Int):
        var pca = PCA(n_components=2)
        var data_2d = pca.fit_transform(data)
        var centroids_2d = pca.transform(centroids.T)
        plt.clf()
        plt.title(f'Iteration {iteration}')
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', marker='o')
        plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x')
        plt.draw()
        plt.pause(0.5)

    var max_iterations = 100
    var k = 3

    # Measure start time
    var start_time = time.time()

    var centroids = random_centroid(data, k)
    var old_centroids = pd.DataFrame()
    var iteration = 1

    # Plot setup
    plt.ion()  # Interactive mode

    # Main loop
    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids.copy()
        var labels = get_labels(data, centroids)
        centroids = new_centroids(data, labels, k)
        plot_clusters(data, labels, centroids, iteration)
        iteration += 1

    # Measure end time
    var end_time = time.time()

    # Calculate the execution time
    var execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")

    # Save plot
    plt.savefig('clusters.png')

if __name__ == "__main__":
    main()