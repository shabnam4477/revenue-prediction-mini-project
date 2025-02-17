
# Data Observation
Dataset is a open banking transaction data it contains 604 transaction rows with 7 features, 2 numerical column and 5 categorical column.
- There are 32 duplicate rows in the dataset, which cause the model to bias to that specific data

# Problem Definition
We are trying to build a revenue prediction model, the problem is regression and we need to use the machine learning model that suitable for this dataset and the problem. Based on experience and the researches, the candidates models are LinearRegression, RandomForrest and XDBoost.

# Data prepration and Feature selection
## Handle missing values
Out of 604 samples, 30 of them are null in the the transaction_amount column, which is approximately the 5% of the samples.
transaction_amount is our target column in this regression problem, what is the best way to fill the null values?

**thaughts**:
- drop the null column

  impossible, because the transaction_amount is our target column
- drop the sample with null values

  considering our dataset size is limited, we prefer to not loose the data
- fill with mean, median

  we have outlier in our dataset, mean and median is so biased to outliers. fill the null values with mean and median cause to inconsistent data
- fill with random value

    since the missing values are in the target column, random guess may cause the bad model performance
- Impute the missing value using KNN

   imputing with KNN Imputer, causes data leakage if the data has not been splitted before imputing
- Analyse the data and research more then choose the proper method

  Target variable is not advised to be imputed, this is because they control how the learning algorithm learns

**Decision**

  5% is not a high percentage so we decide to remove the samples with missing values for now, based on the model result we can come back and make different decision
  robust models like Random forest/LGM/XGBoost will automatically handle missing values but not the target

## Handle outliers
BY visualiting the data in numerical columns, transaction_amount and customer_id. There are some values which are not in the range, we need to replace those values with the value which is in the range
Outlier samples cause underfitting for our model
## Categorical columns
Check the cardinality of the categorical columns and drop the one which have high cardinality. replace categorical features with numerical values 


# Evaluation Metrics
## Mean Absolute Percentage Error (MAPE)
Suitable for the regression problem and calculate the error of predicted revenue vs actual revenue
## Symmetric Mean Absolute Percentage Error (sMAPE)
Since we have both negative and positive revenue, this metric is usefull to calculate the error considering the sign of the data

# Result improvement
Machine learning pipeline is a cycle, to improve the result we need to back to the prev stages and try different approaches.

## Data prespective
- Pick the diffrenet approach to fill the null values for missing data(fill with median and mean)
- Pick different columns as features(use transaction_date column)
- Use different range of parameters for models. The steps are changing the range, monitor the performance 
- Use different method for tune hyper parameters, we currently use GridSearchCV to tune hyperparameters, we can also try RandomizedSearchCV

# Tune hyperparameters
Using RandomizedSearchCV and GridSearchCV methods for tune hyperparameters

In the first run the score of random forrest using RandomizedSearchCV, was 0.5754971243809461, changed it to GridSearchCV, improved the score to 0.61
### GridSearchCV
GridSearchCV can thoroughly evaluate all possible combinations of hyperparameters, which is feasible given the fewer hyperparameters in RandomForest.
### RandomizedSearchCV
Efficiency: Given the potentially large number of hyperparameters, RandomizedSearchCV can help you quickly identify a good set of hyperparameters without needing to evaluate every possible combination.


# Result Observation
## Data analyse
Analysing the predicted revenue in detail can be helpfull, how model perform for each individual company? how model perform predicting negative revenue? how model perform predicting positive revenue?
the answer to above questions helps us to improve the performance by choosing different approach in data prepration or chosing the models or validation metrics
## Model selection
### XGBRegressor
XGBoost is known for its scalability and performance on large datasets, making it a good fit for complex data, not a good choice for our dataset

XGBoost result has overfitting, the train score is far from test score, because of the complexity of the model it works good in the large dataset with more samples and more features. However, our dataset with 604 samples and 7 features cause the model to overfit since it didn't see enough variety of samples in learning process.

Automatic feature selection was useful to improve the score of the linear regression but decrease the score of the random forrest and xdboost, since they need more features to learn and have good performance.

### RandomForestRegressor

