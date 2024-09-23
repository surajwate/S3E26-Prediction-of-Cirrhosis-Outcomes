import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from pathlib import Path
from model_dispatcher import models
import typer
from sklearn.metrics import log_loss
import structlog
import time

# Setup logger
log_path = Path("logs/train.log")
log_path.parent.mkdir(exist_ok=True, parents=True)

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.WriteLoggerFactory(file=log_path.open("a")),
)
logger = structlog.get_logger()


def run(fold, model):

    # Load the training data
    df = pd.read_csv("./input/train_folds.csv")

    # Split into train and test
    train = df[df.kfold != fold].reset_index(drop=True)
    test = df[df.kfold == fold].reset_index(drop=True)

    train_X = train.drop(["id", "Status", "kfold"], axis=1)
    test_X = test.drop(["id", "Status", "kfold"], axis=1)

    train_y = train.Status
    test_y = test.Status

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
    test_y = le.transform(test_y)

    train_y = train_y.ravel()
    test_y = test_y.ravel()


    start = time.time()

    # Fit the model on training data
    try:
        model.fit(train_X, train_y)
    except Exception as e:
        logger.error("Error during model training", fold=fold, error=str(e))
        return

    # Predict probabilities for test data
    try:
        preds = model.predict_proba(test_X)
    except Exception as e:
        logger.error("Error during prediction", fold=fold, error=str(e))
        return

    end = time.time()

    # inverse transform the predictions
    # preds = le.inverse_transform(preds)

    # Calculate log loss
    loss = log_loss(test_y, preds)
    print(f"Fold={fold}, Log loss={loss:.4f}")
    logger.info("Run completed", fold=fold, log_loss=loss, time_taken=f"{end - start:.2f}s")


def main(model):
    for fold_ in range(5):
        run(fold_, model)


if __name__ == "__main__":
    # for model_name, model in models.items():
    #     logger.info(f"Starting training for model: {model_name}")
    #     main(model)
    #     logger.info(f"Training completed for model: {model_name}")
    #     logger.info("=====================================")
    main(models["logistic_regression"])
    
