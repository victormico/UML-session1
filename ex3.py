import numpy as np
from kmeans_class import Kmeans
from sklearn import datasets

X = datasets.load_iris().data

km1 = Kmeans(X, K=1)
km1.set_centers(np.array([X[0, :]]))

for i, x in enumerate(X):
    km1.sequential_step(x, 1 / (i + 1))
km1.plot("K = 1")

km2 = Kmeans(X, K=2)
S0 = X[(0, 1), :]
km2.set_centers(S0)
km2.assignment_step()
for i, x in enumerate(X):
    km2.sequential_step(x, 1 / (i + 1))
km2.assignment_step()
for i, x in enumerate(X):
    km2.sequential_step(x, 1 / (i + 1))
km2.assignment_step()
km2.plot("K = 2")

km3 = Kmeans(X, K=3)
S0 = X[(0, 1, 2), :]
km3.set_centers(S0)
km3.assignment_step()
for i, x in enumerate(X):
    km3.sequential_step(x, 1 / (i + 1))
km3.assignment_step()
for i, x in enumerate(X):
    km3.sequential_step(x, 1 / (i + 1))
km3.assignment_step()
km3.plot("K = 3")
