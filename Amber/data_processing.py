import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

#Three methods of handling missing values
# Load the dataset
train_df = pd.read_csv('../train.csv')
train_df = train_df.fillna(np.nan)

# Simulate missing values in 5% of the dataset
for col in train_df.columns:
    if col != 'Status':
        train_df.loc[train_df.sample(frac=0.05).index, col] = pd.NA

train_df = train_df.fillna(np.nan)
# Divide data into features and target
X = train_df.drop('Status', axis=1)
y = train_df['Status']

# Splitting the dataset into the Training set and Validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling categorical variables and scaling numerical variables
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())  # Scale data
])


numerical_transformer_median = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Interpolation of missing values using median
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One hot encode categorical variables
])



class RandomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.fill_value = {}

    def fit(self, X, y=None):
        for column in X:
            if X[column].dtype == 'object' and X[column].isnull().any():
                self.fill_value[column] = X[column].dropna().sample(1).values[0]
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column in self.fill_value:
            X_copy[column] = X_copy[column].fillna(self.fill_value[column])
        return X_copy

categorical_transformer_random = Pipeline(steps=[
    ('imputer', RandomImputer()),  # Use random values to interpolate missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data - 1
preprocessor_1 = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Bundle preprocessing for numerical and categorical data - 2
preprocessor_2 = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_median, numerical_cols),
        ('cat', categorical_transformer_random, categorical_cols)
    ])


numerical_transformer_iterative = Pipeline(steps=[
    ('imputer', IterativeImputer(estimator=RandomForestRegressor(), max_iter=3, random_state=0)),  # # Iterative interpolation using random forests as estimators
    ('scaler', StandardScaler())
])

# # applies iterative interpolation to numeric features
# preprocessor_3 = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer_iterative, numerical_cols),
#         ('cat', categorical_transformer, categorical_cols)
#     ])
#


# numerical feature -The mean value of each numerical feature column is used to fill in the missing values in that column.
# Category - Fill the missing values in each categorical feature column with the value that occurs most frequently in that column.
X_train_prepared_1 = preprocessor_1.fit_transform(X_train)
X_val_prepared_1 = preprocessor_1.transform(X_val)

# numerical feature - Impute missing values with median
# Category - Use random values to interpolate missing values
X_train_prepared_2 = preprocessor_2.fit_transform(X_train)
X_val_prepared_2 = preprocessor_2.transform(X_val)

#numerical feature - Iterative interpolation
#Category - Impute missing values with the most frequent
# too slow
# X_train_prepared_3 = preprocessor_3.fit_transform(X_train)
# X_val_prepared_3 = preprocessor_3.transform(X_val)
