import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_path = 'AmesHousing.csv'
data = pd.read_csv(data_path)

# Separate target from predictors
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1)

# Split data into train and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)

# Select categorical columns with relatively low cardinality
categorical_cols = [cname for cname in train_X.columns if
                    train_X[cname].nunique() < 10 and train_X[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in train_X.columns if
                  train_X[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Create and evaluate the pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Define GridSearchCV
param_grid = {
    'model__n_estimators': [300, 600, 900],
    'model__max_depth': [None, 15, 20, 25],
    'model__min_samples_split': [2, 10, 20],
    'model__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='neg_mean_absolute_error', verbose=1)
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
