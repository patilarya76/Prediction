"""
evaluate_model.py
-----------------
Loads trained models and test data, calculates comprehensive evaluation metrics
(accuracy, precision, recall, f1-score, ROC AUC, confusion matrix),
extracts feature importance, generates visualization artifacts, and saves
the final evaluation report.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)


def evaluate_models(models_dir: str, figures_dir: str):
    """
    Computes test metrics for Logistic Regression and Random Forest.
    Saves visual artifacts and JSON/CSV metric tables.
    """
    os.makedirs(figures_dir, exist_ok=True)

    lr_path = os.path.join(models_dir, "logistic_regression.joblib")
    rf_path = os.path.join(models_dir, "random_forest.joblib")
    test_path = os.path.join(models_dir, "test_data.joblib")

    if not (os.path.exists(lr_path) and os.path.exists(rf_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            "Trained models or test set not found. Run src/train_model.py first!"
        )

    lr_pipeline = joblib.load(lr_path)
    rf_pipeline = joblib.load(rf_path)
    X_test, y_test = joblib.load(test_path)

    models = {
        'Logistic Regression': lr_pipeline,
        'Random Forest': rf_pipeline
    }

    metrics_records = []
    roc_data = {}
    cm_dict = {}

    print("=" * 65)
    print("MODEL EVALUATION ON TEST SET (20% Holdout, N = {})".format(len(y_test)))
    print("=" * 65)

    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)
        cm_dict[name] = cm

        metrics_records.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4),
            'TN': int(cm[0, 0]),
            'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0]),
            'TP': int(cm[1, 1])
        })

        print(f"\n--- {name.upper()} ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print("\nConfusion Matrix:")
        print(f"  [TN={cm[0,0]}  FP={cm[0,1]}]")
        print(f"  [FN={cm[1,0]}  TP={cm[1,1]}]")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    comparison_df = pd.DataFrame(metrics_records)
    print("=" * 65)
    print("SUMMARY COMPARISON TABLE:")
    print(comparison_df.to_string(index=False))
    print("=" * 65)

    # Save metrics to CSV and JSON
    comparison_csv = os.path.join(models_dir, "model_comparison.csv")
    comparison_json = os.path.join(models_dir, "model_comparison.json")
    comparison_df.to_csv(comparison_csv, index=False)
    with open(comparison_json, 'w') as f:
        json.dump(metrics_records, f, indent=4)

    # 1. Plot Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, cm) in zip(axes, cm_dict.items()):
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Perished (0)', 'Survived (1)'],
            yticklabels=['Perished (0)', 'Survived (1)'],
            ax=ax, annot_kws={'size': 14, 'weight': 'bold'}
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    cm_plot_path = os.path.join(figures_dir, "confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrices to: {cm_plot_path}")

    # 2. Plot ROC Curves
    plt.figure(figsize=(7, 5.5))
    colors = ['#1f77b4', '#2ca02c']
    for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, linewidth=2.2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC = 0.50)')
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.title('ROC Curves: Logistic Regression vs Random Forest', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    roc_plot_path = os.path.join(figures_dir, "roc_curves.png")
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to: {roc_plot_path}")

    # 3. Feature Importance Extraction for Random Forest
    rf_classifier = rf_pipeline.named_steps['classifier']
    preprocessor = rf_pipeline.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()
    # Clean feature names (remove transformer prefixes like num__ and cat__)
    clean_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]

    importances = rf_classifier.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': clean_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fi_csv = os.path.join(models_dir, "feature_importance.csv")
    fi_df.to_csv(fi_csv, index=False)

    # Plot top features
    plt.figure(figsize=(8, 6))
    top_fi = fi_df.head(10).sort_values(by='Importance', ascending=True)
    bars = plt.barh(top_fi['Feature'], top_fi['Importance'], color='#2b5c8f', edgecolor='black', alpha=0.85)
    plt.xlabel('Gini Feature Importance (MDI)', fontsize=11)
    plt.title('Top 10 Feature Importances - Random Forest', fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}",
                 va='center', fontsize=9, fontweight='semibold')
    plt.tight_layout()
    fi_plot_path = os.path.join(figures_dir, "rf_feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot to: {fi_plot_path}")

    # Also extract Logistic Regression Coefficients
    lr_classifier = lr_pipeline.named_steps['classifier']
    coefs = lr_classifier.coef_[0]
    lr_coef_df = pd.DataFrame({
        'Feature': clean_names,
        'Coefficient': coefs,
        'OddsRatio': np.exp(coefs)
    }).sort_values(by='Coefficient', ascending=False)
    lr_coef_df.to_csv(os.path.join(models_dir, "lr_coefficients.csv"), index=False)

    print("\nTop 5 Positive Predictors (Logistic Regression):")
    print(lr_coef_df.head(5)[['Feature', 'Coefficient', 'OddsRatio']].to_string(index=False))
    print("\nTop 5 Negative Predictors (Logistic Regression):")
    print(lr_coef_df.tail(5)[['Feature', 'Coefficient', 'OddsRatio']].to_string(index=False))

    return comparison_df, fi_df, lr_coef_df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_folder = os.path.join(current_dir, "..", "models")
    figures_folder = os.path.join(current_dir, "..", "models")
    evaluate_models(models_folder, figures_folder)
