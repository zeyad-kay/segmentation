# TODO:
# using a feature vector X e.g. RGB, HSV, ... and a window size
# for each point:
#   pick as the center or mean
#   estimate kde of the local neighbourhood
#   now we computed the mean shift vector or the step by which we need to move
#   update the mean by adding to it the mean shift vector
#   repeat until convergence

import numpy as np
from utils import euclidean_distances, kernel_density_estimation
from matplotlib import pyplot as plt


class MeanShift:
    def __init__(self, bandwidth: float, window_size:float = 1,tolerance: float = 1e-2,verbose:bool = False) -> None:
        self.window_size = window_size
        self.bandwidth = bandwidth
        self.tolerance = tolerance
        self.verbose = verbose

    def __check_params(self):
        if self.window_size < 0:
            raise ValueError("window_size must be greater than 0")
        if self.tolerance < 0:
            raise ValueError("tolerance must be greater than 0")
        if self.bandwidth < 0:
            raise ValueError("bandwidth must be greater than 0")

    def fit(self, X: np.ndarray):
        self.__check_params()
        self.centers = np.ndarray((0,X.shape[1]))
        for x in X:
            center = x.copy()
            center = np.array([1,1])
            old_center = np.inf
            while not np.allclose(old_center,center,rtol=self.tolerance,atol=0):
                old_center = center.copy()
                neighbours = self.__get_neighbours(X, center)
                kde = kernel_density_estimation(neighbours - center)
                center = np.sum(kde * neighbours,axis=0)/np.sum(kde,axis=0)
            # if np.abs()
            self.centers = np.append(self.centers, [center],axis=0)
        return self

    def __get_neighbours(self, X: np.ndarray, center: np.ndarray) -> np.ndarray:
        distances = euclidean_distances(X, center)
        return X[distances <= self.window_size]



# from sklearn.cluster import MeanShift
from sklearn.datasets._samples_generator import make_blobs
# from mpl_toolkits.mplot3d import Axes3D

# # We will be using the make_blobs method
# # in order to generate our own data.

clusters = [[2, 2],[7,7],[5,5]]

X, _ = make_blobs(n_samples = 150, centers = clusters,
								cluster_std = 0.60)

# # After training the model, We store the
# # coordinates for the cluster centers
ms = MeanShift(1)
ms.fit(X)
cluster_centers = ms.centers

# Finally We plot the data points
# and centroids in a 3D graph.
fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(X[:, 0], X[:, 1], marker ='o')
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker ='x', color ='red',
		s = 300, linewidth = 5, zorder = 10)

plt.show()
