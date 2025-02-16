from models.base_model import BaseModel
from sklearn.linear_model import LinearRegression


class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(LinearRegression())
