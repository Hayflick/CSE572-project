# Author: Aria Salehi
# Class: CSE 572
# Date: 4/21/24

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and separate predictors from target
data_path = 'AmesHousing.csv'
data = pd.read_csv(data_path)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1)

# Split data into train and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)

# Data exploration - distribution graphs and stuff
plt.scatter(val_X["Year Built"], val_y, color = "teal", alpha=.8)
plt.title("Sale Price vs Year Built")
plt.xlabel("Year Built")
plt.ylabel("Sale Price")
plt.show()

# Select categorical columns with small card (<10)
categorical_cols = [cname for cname in train_X.columns if
                    train_X[cname].nunique() < 10 and train_X[cname].dtype == "object"]

# Select numerical/catagorical columns and preprocess
numerical_cols = [cname for cname in train_X.columns if
                  train_X[cname].dtype in ['int64', 'float64']]
numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model - at first we use random forest
model = RandomForestRegressor(n_estimators=100, random_state=0)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Define the parameter space for grid search
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 15, 20, 25],
    'model__min_samples_split': [2, 10, 20],
    'model__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(train_X, train_y)
print("Best parameters:", grid_search.best_params_)
print("Best CV score:", -grid_search.best_score_)


# Prediction time!
predictions = grid_search.predict(val_X)
print("Mean Absolute Error:", mean_absolute_error(val_y, predictions))

# Plot the feature importance (review execution)
importances = clf.named_steps['model'].feature_importances_
features = numerical_cols + \
           [col for col_name in categorical_cols for col in clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names([col_name])]
feature_importances = pd.DataFrame(importances, index=features, columns=['Importance']).sort_values('Importance', ascending=False)
sns.barplot(x=feature_importances.Importance, y=feature_importances.index)
plt.title('Feature Importances')
plt.show()

# Now repeat the same process for gradient boosting
model = GradientBoostingRegressor(random_state=0)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])
# Define parameter space for grid searching (more tuning needed?)
# Note for later: Evaluation takes too long when going above 300 estimators
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 4, 5],
    'model__min_samples_split': [2, 4],
    'model__min_samples_leaf': [1, 2, 3]
}

grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(train_X, train_y)
print("Best parameters:", grid_search.best_params_)
print("Best CV score:", -grid_search.best_score_)

# Get predictions
predictions = grid_search.predict(val_X)
print("Mean Absolute Error:", mean_absolute_error(val_y, predictions))

# Plotting feature importance
importances = clf.named_steps['model'].feature_importances_
features = numerical_cols + \
           [col for col_name in categorical_cols for col in clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names([col_name])]
feature_importances = pd.DataFrame(importances, index=features, columns=['Importance']).sort_values('Importance', ascending=False)
sns.barplot(x=feature_importances.Importance, y=feature_importances.index)
plt.title('Feature Importances')
plt.show()