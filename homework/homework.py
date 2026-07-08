import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# Paso 1: Limpieza de los datasets
train = pd.read_csv("files/input/train_data.csv.zip")
test = pd.read_csv("files/input/test_data.csv.zip")

train = train.rename(columns={"default payment next month": "default"})
test = test.rename(columns={"default payment next month": "default"})

train = train.drop(columns=["ID"])
test = test.drop(columns=["ID"])

train = train.dropna()
test = test.dropna()

train = train[(train["EDUCATION"] != 0) & (train["MARRIAGE"] != 0)]
test = test[(test["EDUCATION"] != 0) & (test["MARRIAGE"] != 0)]

train["EDUCATION"] = train["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
test["EDUCATION"] = test["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

# Paso 2: Dividir en x_train, y_train, x_test, y_test
x_train = train.drop(columns=["default"])
y_train = train["default"]
x_test = test.drop(columns=["default"])
y_test = test["default"]

# Paso 3: Pipeline para el modelo de clasificación
categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
numeric_cols = [col for col in x_train.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", MinMaxScaler(), numeric_cols),
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("select", SelectKBest(f_classif)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Paso 4: Optimización de hiperparámetros con validación cruzada
param_grid = {
    "select__k": [5],
    "classifier__C": [1, 10],
    "classifier__class_weight": [
        {0: 1, 1: 1.3},
        {0: 1, 1: 1.32},
    ],
}

model = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
)

model.fit(x_train, y_train)

# Paso 5: Guardar el modelo comprimido con gzip
os.makedirs("files/models", exist_ok=True)
with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(model, f)

# Paso 6 y 7: Calcular métricas y matrices de confusión
os.makedirs("files/output", exist_ok=True)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

metrics = []
for dataset_name, y_true, y_pred in [("train", y_train, y_train_pred), ("test", y_test, y_test_pred)]:
    metrics.append({
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    })

for dataset_name, y_true, y_pred in [("train", y_train, y_train_pred), ("test", y_test, y_test_pred)]:
    cm = confusion_matrix(y_true, y_pred)
    metrics.append({
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    })

with open("files/output/metrics.json", "w") as f:
    for entry in metrics:
        f.write(json.dumps(entry) + "\n")