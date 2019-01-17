import concurrent.futures
import logging
import multiprocessing
import numpy as np
import os
from bico.core import BICO
from bico.geometry.point import Point
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

n_samples = 100000
random_state = 170

X_varied, y_varied = make_blobs(n_samples=n_samples, centers=5,
                                cluster_std=[1.0, 3, 0.3, 2, 5],
                                random_state=random_state)

y_pred = KMeans(n_clusters=5, random_state=random_state).fit_predict(X_varied)

projections = 5

proj_method = 'simple'  # binary_tree, binary, simple


def run_bico_parellel(X):
    run_bico(200, X)


def run_bico(size, X):
    bico = BICO(2, n_samples, projections, size, proj_method, False)

    for row in X:
        bico.insert_point(Point(row))

    c = bico.get_coreset()

    for i, time in enumerate(bico.time):
        print("Time spent on level {}: {}".format(i, time))


print(multiprocessing.cpu_count())
print(os.cpu_count())

tstart = datetime.now()
with concurrent.futures.ProcessPoolExecutor() as executor:
    X_chunks = np.array_split(X_varied, 4)
    executor.map(run_bico_parellel, X_chunks)
tend = datetime.now()
print("Time spent: {}".format(tend - tstart))
