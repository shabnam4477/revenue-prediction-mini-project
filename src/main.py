import pandas as pd
import logging

from sklearn.model_selection import train_test_split

from models.linear_regression import LinearRegressionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

OB_df = pd.read_csv("data/processed/clean_revenue_data.csv")

# split data for train and test
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

#LinearRegressionModel
model = LinearRegressionModel()
model.train(X_train, y_train)
mse = model.evaluate(X_test, y_test)
cv_score = model.cross_validate(X, target)
model_score = model.calculate_model_score(X_test, y_test)
logging.info('LinearRegressionModel results')
logging.info(f'MSE is {mse}')
logging.info(f'cross validation mse is {cv_score}')
logging.info(f'model score is {model_score}')


