from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from models.base_model import BaseModel


class RandomForestRegressorModel(BaseModel):
    def __init__(self, model):
        super().__init__(model)

    def tune_hyperparameters(parameters, X_train, y_train):
        rf = RandomForestRegressor()
        grid_search = GridSearchCV(rf, param_grid=parameters)
        grid_search.fit(X_train, y_train)
        return grid_search
