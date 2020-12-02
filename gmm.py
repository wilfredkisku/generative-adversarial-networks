import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets, mixture
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix

plt.rcParams["axes.grid"] = False

n_samples = 100

# define the mean points for each of the systhetic cluster centers
t_means = [[8.4, 8.2], [1.4, 1.6], [2.4, 5.4], [6.4, 2.4]]

# for each cluster center, create a Positive semidefinite convariance matrix
t_covs = []
for s in range(len(t_means)):
  t_covs.append(make_spd_matrix(2))

X = []
for mean, cov in zip(t_means,t_covs):
  x = np.random.multivariate_normal(mean, cov, n_samples)
  X += list(x)

X = np.array(X)
np.random.shuffle(X)
print("Dataset shape:", X.shape)

plt.scatter(X[:,0], X[:,1], marker='.', color='r', linewidth=1)
plt.show()
