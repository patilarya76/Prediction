"""
train_model.py
--------------
Trains Logistic Regression and Random Forest models on Titanic dataset using
scikit-learn Pipelines to eliminate data leakage.

Saves trained models, test sets, and cross-validation metrics.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from data_preprocessing import prepare_dataset, build_preprocessor


def train_models(data_path: str, models_dir: str):
    """
    Executes end-to-end model training:
    1. Loads dataset and extracts engineered features
    2. Performs stratified train/test split (80/20)
    3. Builds ML pipelines (Preprocessor + Classifier)
    4. Evaluates 5-fold Stratified Cross-Validation on training data
    5. Fits models on full training set
    6. Persists models and test splits to disk
    """
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading data and performing feature engineering...")
    X, y, full_df = prepare_dataset(data_path)
    print(f"Total samples: {len(X)}, Features: {list(X.columns)}")

    print("\nSTEP 2: Splitting dataset (80% train, 20% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    print(f"Train survival rate: {y_train.mean():.4f} | Test survival rate: {y_test.mean():.4f}")

    # Build preprocessing pipeline
    preprocessor = build_preprocessor()

    # Define Model Pipelines
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, C=1.0))
    ])

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ))
    ])

    models = {
        'Logistic Regression': lr_pipeline,
        'Random Forest': rf_pipeline
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    print("\nSTEP 3: Cross-validation on training data (5-Fold Stratified CV)...")
    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        cv_results[name] = {
            'mean_cv_accuracy': float(scores.mean()),
            'std_cv_accuracy': float(scores.std())
        }
        print(f"  > {name:20s}: Mean CV Accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")

    print("\nSTEP 4: Fitting final models on full training set...")
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        print(f"  > Fitted: {name}")

    print("\nSTEP 5: Saving artifacts to disk...")
    joblib.dump(lr_pipeline, os.path.join(models_dir, "logistic_regression.joblib"))
    joblib.dump(rf_pipeline, os.path.join(models_dir, "random_forest.joblib"))
    
    # Save test set for clean standalone evaluation
    joblib.dump((X_test, y_test), os.path.join(models_dir, "test_data.joblib"))
    joblib.dump((X_train, y_train), os.path.join(models_dir, "train_data.joblib"))
    joblib.dump(cv_results, os.path.join(models_dir, "cv_results.joblib"))

    print(f"All artifacts saved in: {models_dir}")
    print("=" * 60)

    return models, (X_train, X_test, y_train, y_test), cv_results


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "..", "data", "titanic.csv")
    models_folder = os.path.join(current_dir, "..", "models")
    train_models(data_file, models_folder)
