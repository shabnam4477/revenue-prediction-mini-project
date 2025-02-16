from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np


class BaseModel:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

    def cross_validate(self, X, y, cv=5):
        scores = -1 * cross_val_score(
            self.model, X, y, cv=cv, scoring="neg_mean_squared_error"
        )
        mean_score = np.mean(scores)
        return mean_score
    
    def calculate_model_score(self, X_test, y_test):
        return self.model.score(X_test, y_test)
