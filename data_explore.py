import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#data explore

# Load the dataset
train_df = pd.read_csv('train.csv')

# Display basic information and the first few rows of the dataset
print(train_df.info())
print(train_df.head())

# Summary statistics for numerical features
print(train_df.describe())

# Summary statistics for categorical features
print(train_df.describe(include=['O']))

# Histograms for numerical features
train_df.hist(bins=15, figsize=(15, 10))
plt.show()

# Boxplots for numerical features by 'Status'
for column in train_df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Status', y=column, data=train_df)
    plt.title(f'Boxplot of {column} by Status')
    plt.show()

# Pairplot to observe pairwise relationships and distributions
sns.pairplot(train_df, hue='Status')
plt.show()

# Simulate missing values in 5% of the dataset
for col in train_df.columns:
    train_df.loc[train_df.sample(frac=0.05).index, col] = pd.NA
