first run random forrest with tume param, the Mean Squared Error: 1244270.5725135717
Model Score: 0.5754971243809461

# Feature selection
This is useful to improve the score of the linear regression but decrease the score of the random forrest and xdboost

# Tune hyperparameters
based on the score you get using the best parameters, you can change the range of parameters to get the better result
## XGBRegressor with RandomizedSearchCV
Efficiency: Given the potentially large number of hyperparameters, RandomizedSearchCV can help you quickly identify a good set of hyperparameters without needing to evaluate every possible combination.
Scalability: XGBoost is known for its scalability and performance on large datasets, making it a good fit for complex data like open banking transactions.

## RandomForestRegressor with GridSearchCV
Thoroughness: GridSearchCV can thoroughly evaluate all possible combinations of hyperparameters, which is feasible given the fewer hyperparameters in RandomForest.
Interpretability: Random forests are generally easier to interpret, which can be beneficial when explaining the model's predictions based on transaction data.

# Evaluation Metrics
## Mean Absolute Percentage Error (MAPE)
## Symmetric Mean Absolute Percentage Error (sMAPE)

# Result Observation
XGBoost result has overfitting, the train score is higher than test score