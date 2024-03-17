from data_processing import preprocessor_1, X_train, X_val,X_train_prepared_1,X_val_prepared_1,y_train,y_val,train_df
from data_processing import X_train_prepared_2,X_val_prepared_2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pandas as pd
# from data_balance import X_balanced,y_balanced
from sklearn.feature_selection import SelectFromModel

# different proportion of status
status_proportions = train_df['Status'].value_counts(normalize=True)

print(status_proportions)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_prepared_1, y_train)


# Predict and evaluate the model
predictions = model.predict(X_val_prepared_1)
accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy}')
print(classification_report(y_val, predictions))
print(confusion_matrix(y_val, predictions))

# #use balanced data
#
# model_bal_1 = RandomForestClassifier(n_estimators=100, random_state=42)
# model_bal_1.fit(X_balanced,y_balanced)
#
# # Predict and evaluate the model
# predictions = model_bal_1.predict(X_val_prepared_1)
# y_val, unique = pd.factorize(y_val)
# accuracy = accuracy_score(y_val, predictions)
# print(f'Accuracy: {accuracy}')
# print(classification_report(y_val, predictions))
# print(confusion_matrix(y_val, predictions))

# Train the model
model_2 = RandomForestClassifier(class_weight='balanced',n_estimators=100, random_state=42)
model_2.fit(X_train_prepared_1, y_train)


# Predict and evaluate the model
predictions = model_2.predict(X_val_prepared_1)
accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy}')
print(classification_report(y_val, predictions))
print(confusion_matrix(y_val, predictions))




# Train the model
model_3 = RandomForestClassifier(class_weight='balanced',n_estimators=100, random_state=42)
model_3.fit(X_train_prepared_2, y_train)


# Predict and evaluate the model
predictions = model_3.predict(X_val_prepared_2)
accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy}')
print(classification_report(y_val, predictions))
print(confusion_matrix(y_val, predictions))

# select import features
selector = SelectFromModel(model_3, prefit=True)
X_train_selected = selector.transform(X_train_prepared_2)
X_val_selected = selector.transform(X_val_prepared_2)


model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
model.fit(X_train_selected, y_train)

# use selected features to predict
predictions = model.predict(X_val_selected)
accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy}')
print(classification_report(y_val, predictions))
print(confusion_matrix(y_val, predictions))