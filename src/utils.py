from math import sqrt, pi
from matplotlib import pyplot as plt
import numpy as np

def kernel_density_estimation(X:np.ndarray, kernel:str="gaussian", bandwidth: float=1):
    if kernel != "gaussian":
        raise NotImplementedError(f"'{kernel}' kernel hasn't been implemented yet. Use 'gaussian' instead")
    
    return 1/(bandwidth*sqrt(2*pi)) * np.exp(-0.5 * (X/bandwidth)**2)

def euclidean_distances(x1, x2):
    return np.linalg.norm(x1 - x2, ord=2, axis=1)


# X=np.random.normal(2.5, 0.9, 14)
# plt.hist(X,4)
# plt.show()
# k = kde(X-[0.8])
# print(k)
# print(sum(k)/k.shape[0])
# plt.hist(k,4)
# plt.show()

# def gaussian(x,b=1):
#     return np.exp(-x**2/(2*b**2))/(b*np.sqrt(2*np.pi))

# X=np.random.normal(2.5, 0.9, 14)
# X_plot = np.linspace(-1, 7, 100)[:, None]
# N=X.shape[0]
# sum1=np.zeros(len(X_plot))
# for i in range(0, N):
#     sum1+=((gaussian(X_plot-X[i]))/N)[:,0]

# print(np.mean(sum1))
# plt.plot(X_plot,sum1)
# plt.show()