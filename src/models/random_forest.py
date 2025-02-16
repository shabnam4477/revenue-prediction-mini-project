from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from models.base_model import BaseModel


class RandomForestRegressorModel(BaseModel):
    def __init__(self, hyperparameters):
        super().__init__(RandomForestRegressor(**hyperparameters))

    @staticmethod
    def tune_hyperparameters(parameters, X_train, y_train):
        rf = RandomForestRegressor()
        grid_search = RandomizedSearchCV(
            rf, param_distributions=parameters, n_iter=5, cv=5
        )
        grid_search.fit(X_train, y_train)
        return grid_search
