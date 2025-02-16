import pandas as pd
import logging

from sklearn.model_selection import train_test_split

from models.linear_regression import LinearRegressionModel
from models.random_forest import RandomForestRegressorModel
from models.base_model import BaseModel
from models.xg_boost import XgBoostRegressionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)


def run_pipeline(model: BaseModel, X_train, X_test, y_train, y_test, X, target):
    model.train(X_train, y_train)
    mse = model.evaluate(X_test, y_test)
    cv_mse = model.cross_validate(X, target, scoring="neg_mean_squared_error")
    cv_score = model.cross_validate(X, target)
    model_score = model.calculate_model_score(X_test, y_test)
    train_score = model.calculate_model_score(X_train, y_train)
    logging.info(f"MSE is {mse}")
    logging.info(f"cross validation mse is {cv_mse}")
    logging.info(f"cross validation score is {cv_score}")
    logging.info(f"model score is {model_score}")
    logging.info(f"model score for training data is {train_score}")


def main():
    # Data prepration
    OB_df = pd.read_csv("data/processed/clean_revenue_data.csv")
    X = OB_df.drop(columns="transaction_amount")
    print(X.head())
    target = OB_df["transaction_amount"]
    print(target.head())
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.2, random_state=42
    )
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # LinearRegressionModel
    logging.info(">>>>> LinearRegression results")
    model = LinearRegressionModel()
    X_train_selected, X_test_selected = model.feature_selection(X_train, y_train, X_test)
    run_pipeline(
        model,
        X_train=X_train_selected,
        y_train=y_train,
        X_test=X_test_selected,
        y_test=y_test,
        X=X,
        target=target,
    )

    # RandomForestRegressor
    logging.info(">>>>> RandomForestRegressor results")
    parameters = {
        "n_estimators": [100, 150, 200, 250, 300],
        "max_depth": range(1, 5),
    }
    grid_search = RandomForestRegressorModel.tune_hyperparameters(
        parameters, X_train, y_train
    )
    best_params = grid_search.best_params_
    model = RandomForestRegressorModel(grid_search.best_estimator_)

    logging.info(f"best params is {best_params}")
    run_pipeline(
        model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X=X,
        target=target,
    )

    # XGBoost
    logging.info(">>>>> XgBoostRegression results")
    parameters = {
        "n_estimators": [10, 20, 30, 40],
        "max_depth": range(1, 5),
        "learning_rate": [0.05, 0.1, 0.2],
    }
    grid_search = XgBoostRegressionModel.tune_hyperparameters(
        parameters, X_train, y_train
    )
    best_params = grid_search.best_params_
    model = XgBoostRegressionModel(best_params)
    logging.info(f"best params is {best_params}")
    run_pipeline(
        model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X=X,
        target=target,
    )


if __name__ == "__main__":
    main()
