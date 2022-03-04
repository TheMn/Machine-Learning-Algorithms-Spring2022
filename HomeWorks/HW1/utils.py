import numpy as np

def mean_normalize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    norm = np.max(X_train, axis=0) - np.min(X_train, axis=0)
    return (X_train - mean)/norm, (X_test - mean)/norm

# import numpy as np
# def mean_normalize(X_train, X_test):
#     maximum = np.max(X_train, axis=0)
#     minimun = np.min(X_train, axis=0)
#     norm = maximum - minimun
#     mean = np.mean(X_train, axis=0)
#     return (X_train - mean)/norm, (X_test - mean)/norm