import sys
import numpy as np

class KMeans():
    def __init__(self, K:int=2, max_iter:int =10, verbose:bool= False) -> None:
        self.K = K
        self.max_iter = max_iter
        self.verbose = verbose

    def __check_params(self):
        if self.K < 2:
            raise ValueError(f"K must be greater than or equal 2")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be greater than or equal 1")
    
    # for each point
    # calculate the distance to each centroid
    # assign the point to cluster k based on least centroid
    # update centroid to mean of points in k
    def fit(self, X: np.ndarray):
        self.__check_params()
        self.__init_centroids(X)
        
        for i in range(self.max_iter):            
            self.labels = self.__assign_labels(X)

            old_centroids = self.centroids.copy()
            self.__update_centroids(X,self.labels)

            if np.array_equal(old_centroids,self.centroids):
                if self.verbose:
                    print(f"converged at iteration #{i+1}")
                return self

            if self.verbose:
                print(f"iteration #{i+1}:")
                [print(f"\tcentroid {j+1}: {k}") for j,k in enumerate(self.centroids)]
        
        return self

    def __update_centroids(self, X: np.ndarray, labels:np.ndarray):
        for k in range(self.K):
            self.centroids[k] = X[np.where(labels==k)[0]].mean(axis=0)

    def __assign_labels(self, X: np.ndarray)-> np.ndarray:
        distances = np.ndarray((self.K,X.shape[0]))
        for i,c in enumerate(self.centroids):
            distances[i] = self.__euclidean_distance(X, c)

        return np.argmin(distances,0)

    def __euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2, ord=2, axis=1)

    def __init_centroids(self, X):
        self.centroids = np.array([X[np.random.randint(0,X.shape[0])] for _ in range(self.K)])
        if self.verbose:
            [print(f"initial centroid {i+1}: {k}") for i,k in enumerate(self.centroids)]

if __name__ == "__main__":
    import cv2
    
    if len(sys.argv) != 2:
        print("Must supply path to image file")

    try:
        img = cv2.imread(sys.argv[1])

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        pixel_vals = image.reshape((-1,3))

        # Convert to float type
        pixel_vals = np.float32(pixel_vals)

        m = KMeans(K=3).fit(pixel_vals)

        # convert data into 8-bit values
        centers = np.uint8(m.centroids)
        segmented_data = centers[m.labels.flatten()]
    
        # reshape data into the original image dimensions
        segmented_image = segmented_data.reshape((image.shape))
    
        cv2.imshow("original",img)
        cv2.imshow("segmented",segmented_image)
        cv2.waitKey(0)
    except cv2.error:
        print("Must supply valid image path")