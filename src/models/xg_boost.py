from sklearn.model_selection import GridSearchCV
from xgboost import XGBRFRegressor, XGBRegressor
from models.base_model import BaseModel


class XgBoostRegressionModel(BaseModel):
    def __init__(self, hyperparameters):
        super().__init__(XGBRegressor(**hyperparameters))

    def tune_hyperparameters(parameters, X_train, y_train):
        xgb_model = XGBRFRegressor()
        random_search = GridSearchCV(xgb_model, param_grid=parameters, cv=5, n_jobs=-1)
        random_search.fit(X_train, y_train)
        return random_search
