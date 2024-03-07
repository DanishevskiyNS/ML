import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score

from utils.plotter import Plotter


def print_f1_scores(y: np.ndarray, hc_clusters: np.ndarray):
    print("F1 micro: ", f1_score(y, hc_clusters, average="micro"))
    print("F1 macro: ", f1_score(y, hc_clusters, average="macro"))
    print("F1 weighted: ", f1_score(y, hc_clusters, average="weighted"))
    print('\n')


def hc_labels(clusters: np.ndarray, label_order: list) -> np.ndarray:
    hc_clusters = [] 

    for v in clusters:
        if v == np.unique(clusters)[0]:
            hc_clusters.append(label_order[0])
        if v == np.unique(clusters)[1]:
            hc_clusters.append(label_order[1])
        if v == np.unique(clusters)[2]:
            hc_clusters.append(label_order[2])

    return np.array(hc_clusters)


def main(path: str):
    # Load data.
    header = [
            "Sepal length", "Sepal width",
            "Petal length", "Petal width",
            "Class"]
    mapping_enc = {"Iris-setosa": 0, 
                   "Iris-versicolor": 1,
                   "Iris-virginica": 2}
    data = pd.read_csv(path, names=header)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].map(mapping_enc).values

    # Hyperparameters.
    hyper = {
        "KMeans": {
            "n_clusters": 3,
            "init": "k-means++", # or "random".
            "n_init": "auto",
            "max_iter": 100, # default 300.
            "tol": 0.1, # default is 0.0001.
            "algorithm": "lloyd", # or "elkan".
            "random_state": 42
        },
        "DBSCAN": {
            "eps": 0.8, 
            "min_samples": 14, 
            "metric": 'l1',
        },
        "Hierarchical": {
            "n_clusters": 3,
            "compute_distances": True,
        }
    }

    # Clusters.
    kmeans = KMeans(**hyper["KMeans"])
    dbscan = DBSCAN(**hyper["DBSCAN"])
    hierarch = AgglomerativeClustering(**hyper["Hierarchical"])

    algorithms = {
        kmeans : {
            "label_order": [2, 0, 1],
            "name": "KMeans"
        },
        dbscan : {
            "label_order": [2, 0, 1],
            "name": "DBSCAN"
        },
        hierarch : {
            "label_order": [1, 0, 2],
            "name": "Hierarchical"
        }
    }

    for algorithm, cfg in algorithms.items():
        clusters = algorithm.fit_predict(X)

        # HARDCODE!
        # The reason of following hardcode is to ensure that cluster's random labels
        # may be compared with actual class labels.
        # E.G. y_true = [0 0 1 1 2 2], clusters = [1 1 2 2 0 0]

        label_order = cfg["label_order"]
        hc_clusters = hc_labels(clusters, label_order)
        print_f1_scores(y, hc_clusters)

        # Plotting 2d scatterplots for very feature pair.
        cl_plotter = Plotter(X, y, hc_clusters, header, cfg["name"])
        cl_plotter.saveall_2d_scatter()

        if cfg["name"] == "Hierarchical":
            model = algorithm.fit(X)
            cl_plotter.plot_dendrogram(model, truncate_mode="level", p=3)
            cl_plotter.save_to_file("../output/Hierarchical/dendrogram.png")


if __name__ == "__main__":
    file_path = "../data/iris.data"
    main(path=file_path)

