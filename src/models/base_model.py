from sklearn.feature_selection import SelectFromModel
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

    def cross_validate(self, X, y, cv=5, scoring=None):
        neg = 1
        if scoring:
            neg = -1
        scores = neg * cross_val_score(self.model, X, y, cv=cv, 
                                       scoring=scoring)
        mean_score = np.mean(scores)
        return mean_score

    def calculate_model_score(self, X_test, y_test):
        return self.model.score(X_test, y_test)
    
    def feature_selection(self, X_train, y_train, X_test):
        feature_selector = SelectFromModel(self.model, threshold='median')
        feature_selector.fit(X_train, y_train)
        X_train_selected = feature_selector.transform(X_train)
        X_test_selected = feature_selector.transform(X_test)
        return X_train_selected, X_test_selected

    @staticmethod
    def tune_hyperparameters():
        pass
