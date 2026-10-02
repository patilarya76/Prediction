"""
data_preprocessing.py
---------------------
Modular preprocessing functions and scikit-learn Pipeline / ColumnTransformer
construction for the Titanic Survival Prediction project.

Author: Data Science Engineering Project
Design: Avoids data leakage by fitting transformers strictly on training folds.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw Titanic passenger dataset from CSV.
    """
    df = pd.read_csv(filepath)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the Titanic dataset:
    - FamilySize = SibSp + Parch + 1 (including the passenger)
    - IsAlone = 1 if FamilySize == 1 else 0
    - Title extraction from passenger Name (categorized to common titles)
    
    Returns a copy of the dataframe with added features.
    """
    df = df.copy()

    # Family Size & IsAlone
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Title extraction from Name
    # E.g. "Braund, Mr. Owen Harris" -> "Mr"
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Consolidate rare titles into standard buckets
    rare_titles = [
        'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
        'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'
    ]
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    df['Title'] = df['Title'].fillna('Rare')

    return df


def get_feature_lists():
    """
    Returns the designated numeric and categorical feature columns
    used for machine learning modeling.
    """
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    categorical_features = ['Sex', 'Pclass', 'Embarked', 'IsAlone', 'Title']
    return numeric_features, categorical_features


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer to handle imputation, scaling,
    and categorical encoding without data leakage.

    Numeric Pipeline:
    - Median imputation (robust to extreme outlier fares/ages)
    - Standard scaling (zero mean, unit variance for Logistic Regression convergence)

    Categorical Pipeline:
    - Most frequent / mode imputation (for missing Embarked values)
    - OneHotEncoder (handle_unknown='ignore', drop='first' where appropriate)
    """
    numeric_features, categorical_features = get_feature_lists()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor


def prepare_dataset(filepath: str):
    """
    End-to-end data loader and feature engineer.
    Returns (X, y, df_engineered).
    """
    raw_df = load_data(filepath)
    df_engineered = engineer_features(raw_df)

    numeric_features, categorical_features = get_feature_lists()
    all_features = numeric_features + categorical_features

    X = df_engineered[all_features].copy()
    y = df_engineered['Survived'].copy()

    return X, y, df_engineered


if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "titanic.csv")
    X, y, df = prepare_dataset(data_path)
    print(f"Dataset successfully prepared. Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Engineered columns sample:\n{X.head(3)}")
