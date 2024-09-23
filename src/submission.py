import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier



# Split into train and test
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

train_X = train.drop(["id", "Status"], axis=1)
test_X = test.drop(["id"], axis=1)

train_y = train.Status

categorical_columns = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
numerical_columns = [
    "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper",
    "Alk_Phos", "SGOT", "Tryglicerides", "Platelets",
    "Prothrombin", "Stage"
]

# One-hot encode categorical features
ohe = OneHotEncoder(sparse_output=False, drop="first")
full_categorical_data = pd.concat([train_X[categorical_columns], test_X[categorical_columns]], axis=0)
ohe.fit(full_categorical_data)
train_ohe = ohe.transform(train_X[categorical_columns])
test_ohe = ohe.transform(test_X[categorical_columns])

# Standardize numerical features
scalar = StandardScaler()
scalar.fit(train_X[numerical_columns])
train_scalar = scalar.transform(train_X[numerical_columns])
test_scalar = scalar.transform(test_X[numerical_columns])

# Combine transformed numerical and categorical features
train_X = np.concatenate([train_ohe, train_scalar], axis=1)
test_X = np.concatenate([test_ohe, test_scalar], axis=1)

# label encode the target variable
le = LabelEncoder()
train_y = le.fit_transform(train_y)

train_y = train_y.ravel()

model = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100, random_state=42))

# Fit the model on training data
model.fit(train_X, train_y)

# Predict probabilities for test data
preds = model.predict_proba(test_X)


# inverse transform the predictions
# preds = le.inverse_transform(preds)

# Calculate log loss
train_preds = model.predict_proba(train_X)
loss = log_loss(train_y, train_preds)
print(f"Log loss={loss:.4f}")

# # Predict class labels based on probabilities
# preds_labels = model.classes_[np.argmax(preds, axis=1)]

# # Inverse transform the predictions
# preds_labels = le.inverse_transform(preds_labels)

submission = pd.read_csv("./input/sample_submission.csv")

submission[['Status_C', 'Status_CL', 'Status_D']] = preds

submission.to_csv("./output/submission.csv", index=False)

