import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

class Plotter(object):
    def __init__(self, X, y1, y2=None, header=None, name=None):
        self.X = X
        self.y1 = y1
        self.y2 = y2
        self.header = header
        self.name = name

    @staticmethod
    def plot_2d_scatter(X, y, ax, name, features, header):
        try:
            label = []
            for _ in y:
                if _ == 0:
                    label.append("purple")
                elif _ == 1:
                    label.append("orange")
                elif _ == 2:
                    label.append("red")
            
            f1_values = X[:, features][:, 0]
            f2_values = X[:, features][:, 1]

            ax.scatter(f1_values, f2_values, c=label)
            ax.set_title(name)
            ax.set_xlabel(header[features[0]])
            ax.set_ylabel(header[features[1]])

        except TypeError as e:
            print("\nNo label data was provided.\n")
            raise e


    @staticmethod 
    def save_to_file(path):
        plt.savefig(path)
        plt.close()


    def horizontal_plots(self, features: tuple):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.tight_layout(pad=5.0)

        name_true = "Y True"
        name_pred = "Predicted Clusters"

        self.plot_2d_scatter(self.X, self.y1, ax1, name_true,
                             features, self.header)
        self.plot_2d_scatter(self.X, self.y2, ax2, name_pred,
                             features, self.header)

        self.save_to_file(f"../output/{self.name}/{self.name}-{features}.png")


    def saveall_2d_scatter(self):
        count_features = self.X.shape[1]

        for f1 in range(count_features):
            for f2 in range(count_features):
                if f1 == f2:
                    continue
                else:
                    self.horizontal_plots((f1, f2))


    def plot_dendrogram(self, model, **kwargs):
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        plt.title("Hierarchical Clustering Dendrogram")
        dendrogram(linkage_matrix, **kwargs)

