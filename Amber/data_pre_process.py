from data_processing import train_df
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from data_processing import preprocessor_1, X_train, X_val,X_train_prepared_1,X_val_prepared_1,y_train,y_val,train_df


encoder = OneHotEncoder(sparse=False)
drug_encoded = pd.DataFrame(encoder.fit_transform(train_df[["Drug"]]), columns=encoder.get_feature_names(["Drug"]))
data = pd.concat([train_df, drug_encoded], axis=1)
data.drop(columns=["Drug"], inplace=True)