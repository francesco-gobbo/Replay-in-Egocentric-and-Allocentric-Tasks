import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression as mi_skl
import multiprocessing
import time


def SI_shuffles(events, vector, iterations = 1000):
    with multiprocessing.Pool() as pool:
        mi_shuf = np.zeros((iterations, events.shape[1]))
        items = [(events, vector) for _ in range(iterations)]
        for s, result in enumerate(pool.starmap(mutual_info, items)):
            mi_shuf[s,:] = result # TODO: this has to be a list of elements not an array element    
    mi_shuf = np.array(mi_shuf) # 2D array (samples x n_cells)
    return mi_shuf

def mutual_info(features, target):
    """ Find mutual inforamtion between a discrete random variable (position of rat) and a set of random variables (events of each cell).
    Function created for parallelizing for lool.
    INPUTS:
    features = 
    target = 
    OUTPUT:
    mi = 
    """
    np.random.shuffle(target)
    mi = mi_skl(features, target, n_neighbors=3)
    return mi