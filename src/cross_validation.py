import numpy as np
from sklearn.model_selection import cross_val_score


def perform_cross_validation(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    mean_score = np.mean(scores)
    return mean_score
