import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# from typing import List
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_models(data: pd.DataFrame) -> pd.DataFrame:

    X = data.drop(['TARGET'], axis=1)
    y = data['TARGET']

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

    for col in categorical_features:
        X[col] = X[col].fillna(X[col].mode().iloc[0])

    label_encoders = {}
    for col in categorical_features:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numerical_features),])

    models = [
        ('Decision Tree', DecisionTreeClassifier(
            random_state=0, class_weight='balanced')),
        ('Random Forest', RandomForestClassifier(
            random_state=0, class_weight='balanced')),
        ('LightGBM', LGBMClassifier(verbosity=0,
         random_state=0, class_weight='balanced')),
        ('XGBoost', XGBClassifier(
            seed=0, scale_pos_weight=np.sqrt((len(y) - sum(y)) / sum(y))))
    ]

    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []

    for name, model in models:
        pipeline = Pipeline(
            steps=[('preprocessor', preprocessor), ('model', model)])
        model_names.append(name)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        roc_aucs.append(roc_auc_score(y_test, y_pred))

    result_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores,
        'ROC AUC': roc_aucs
    })

    return result_df
