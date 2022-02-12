import numpy as np
from kmeans_class import Kmeans
from sklearn import datasets


X = datasets.load_iris().data
np.random.seed(1)
K = 3
km = Kmeans(X, K)

S0 = X[(0, 1, 2), :]

km.set_centers(S0)
km.assignment_step()
km.plot("1st assignment")
km.updating_step().assignment_step()
km.plot("2nd assignment")
km.updating_step().assignment_step()
km.plot("3rd assignment")