from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def calculate_error(y_test, y_predict):
    y_test=np.array(y_test)
    error = []
    for i in range(len(y_test)-1):
        actual = y_test[i]
        predict = y_predict[i]
        both_negative = actual < 0 and predict < 0
        both_positive = actual > 0 and predict > 0
        if both_negative or both_positive:
            diff = abs(abs(actual) - abs(predict))
        else:
            diff = abs(actual) + abs(predict)
        error.append(diff*100/abs(actual))
    return np.mean(error)

class BaseModel:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        return self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return np.round(y_pred, 1)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        MSE = mean_squared_error(y_test, predictions)
        MAPE = mean_absolute_percentage_error(y_test, predictions)
        SMAPE = mean_absolute_percentage_error(y_test, predictions)
        custom_metric =calculate_error(y_test, predictions)
        return MSE, MAPE, SMAPE, custom_metric

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
