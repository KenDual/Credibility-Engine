"""
XGBoost Model Training for E-commerce Return Fraud Detection
Author: Your Name
Date: 2025-10-14
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, 
                             precision_recall_curve, confusion_matrix, 
                             classification_report)
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("XGBOOST MODEL TRAINING - E-COMMERCE RETURN FRAUD DETECTION")
print("="*80)

# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================
print("\n[1] LOADING PROCESSED DATA...")

train_data = pd.read_csv('data/train_data.csv')
val_data = pd.read_csv('data/val_data.csv')
test_data = pd.read_csv('data/test_data.csv')

X_train = train_data.drop('Target', axis=1)
y_train = train_data['Target']

X_val = val_data.drop('Target', axis=1)
y_val = val_data['Target']

X_test = test_data.drop('Target', axis=1)
y_test = test_data['Target']

print(f"✓ Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"✓ Validation set: {X_val.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
print(f"\nClass distribution:")
print(f"  Train: {y_train.sum()} Returned ({y_train.mean()*100:.1f}%)")
print(f"  Val: {y_val.sum()} Returned ({y_val.mean()*100:.1f}%)")
print(f"  Test: {y_test.sum()} Returned ({y_test.mean()*100:.1f}%)")

# ============================================================================
# 2. BASELINE MODEL - LOGISTIC REGRESSION
# ============================================================================
print("\n[2] TRAINING BASELINE MODEL (Logistic Regression)...")

baseline_model = LogisticRegression(random_state=42, max_iter=1000)
baseline_model.fit(X_train, y_train)

# Evaluate baseline
y_val_pred_baseline = baseline_model.predict(X_val)
y_val_proba_baseline = baseline_model.predict_proba(X_val)[:, 1]

baseline_metrics = {
    'accuracy': accuracy_score(y_val, y_val_pred_baseline),
    'precision': precision_score(y_val, y_val_pred_baseline),
    'recall': recall_score(y_val, y_val_pred_baseline),
    'f1': f1_score(y_val, y_val_pred_baseline),
    'auc': roc_auc_score(y_val, y_val_proba_baseline)
}

print(f"\nBaseline Model Performance (Validation Set):")
for metric, value in baseline_metrics.items():
    print(f"  {metric.upper()}: {value:.4f}")

# ============================================================================
# 3. XGBOOST MODEL - INITIAL TRAINING
# ============================================================================
print("\n[3] TRAINING XGBOOST MODEL (Initial)...")

# Initial XGBoost model with default params
xgb_initial = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='logloss'
)

xgb_initial.fit(X_train, y_train, verbose=False)

# Evaluate initial model
y_val_pred_initial = xgb_initial.predict(X_val)
y_val_proba_initial = xgb_initial.predict_proba(X_val)[:, 1]

initial_metrics = {
    'accuracy': accuracy_score(y_val, y_val_pred_initial),
    'precision': precision_score(y_val, y_val_pred_initial),
    'recall': recall_score(y_val, y_val_pred_initial),
    'f1': f1_score(y_val, y_val_pred_initial),
    'auc': roc_auc_score(y_val, y_val_proba_initial)
}

print(f"\nXGBoost Initial Model Performance (Validation Set):")
for metric, value in initial_metrics.items():
    print(f"  {metric.upper()}: {value:.4f}")

print(f"\nImprovement over baseline:")
print(f"  AUC: +{(initial_metrics['auc'] - baseline_metrics['auc'])*100:.2f}%")
print(f"  F1: +{(initial_metrics['f1'] - baseline_metrics['f1'])*100:.2f}%")

# ============================================================================
# 4. HYPERPARAMETER TUNING
# ============================================================================
print("\n[4] HYPERPARAMETER TUNING (This may take 5-10 minutes)...")

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

# GridSearchCV
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss'
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Starting grid search...")
grid_search.fit(X_train, y_train)

# Best model
best_xgb = grid_search.best_estimator_
print(f"\n✓ Grid search completed!")
print(f"\nBest parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest cross-validation AUC: {grid_search.best_score_:.4f}")

# ============================================================================
# 5. EVALUATE BEST MODEL
# ============================================================================
print("\n[5] EVALUATING BEST MODEL...")

# Validation set
y_val_pred = best_xgb.predict(X_val)
y_val_proba = best_xgb.predict_proba(X_val)[:, 1]

# Test set
y_test_pred = best_xgb.predict(X_test)
y_test_proba = best_xgb.predict_proba(X_test)[:, 1]

# Metrics
def calculate_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_proba)
    }

val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

print("\nFinal Model Performance:")
print("\nValidation Set:")
for metric, value in val_metrics.items():
    print(f"  {metric.upper()}: {value:.4f}")

print("\nTest Set:")
for metric, value in test_metrics.items():
    print(f"  {metric.upper()}: {value:.4f}")

# Classification report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['Not Returned', 'Returned']))

# ============================================================================
# 6. CONFUSION MATRIX
# ============================================================================
print("\n[6] GENERATING CONFUSION MATRIX...")

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Returned', 'Returned'],
            yticklabels=['Not Returned', 'Returned'])
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add percentages
for i in range(2):
    for j in range(2):
        pct = cm[i, j] / cm.sum() * 100
        plt.text(j+0.5, i+0.7, f'({pct:.1f}%)', 
                ha='center', va='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('notebooks/10_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/10_confusion_matrix.png")
plt.close()

# ============================================================================
# 7. ROC CURVE
# ============================================================================
print("\n[7] GENERATING ROC CURVE...")

fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
auc_score = roc_auc_score(y_test, y_test_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='#3B82F6', linewidth=2, label=f'XGBoost (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - XGBoost Model', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/11_roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/11_roc_curve.png")
plt.close()

# ============================================================================
# 8. PRECISION-RECALL CURVE
# ============================================================================
print("\n[8] GENERATING PRECISION-RECALL CURVE...")

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_test_proba)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='#EF4444', linewidth=2, label=f'XGBoost (AUC = {auc_score:.4f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve - XGBoost Model', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=11)
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('notebooks/12_precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/12_precision_recall_curve.png")
plt.close()

# ============================================================================
# 9. FEATURE IMPORTANCE
# ============================================================================
print("\n[9] ANALYZING FEATURE IMPORTANCE...")

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_xgb.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# Save full feature importance
feature_importance.to_csv('models/feature_importance.csv', index=False)
print("✓ Saved: models/feature_importance.csv")

# Plot top 20 features
plt.figure(figsize=(12, 8))
top_20 = feature_importance.head(20)
plt.barh(range(len(top_20)), top_20['Importance'].values, color='#8B5CF6')
plt.yticks(range(len(top_20)), top_20['Feature'].values)
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/13_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/13_feature_importance.png")
plt.close()

# ============================================================================
# 10. THRESHOLD CALIBRATION FOR RISK LEVELS
# ============================================================================
print("\n[10] CALIBRATING RISK THRESHOLDS...")

# Define risk thresholds
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70

# Classify into risk levels
def classify_risk(probability):
    if probability < LOW_RISK_THRESHOLD:
        return 'LOW'
    elif probability < HIGH_RISK_THRESHOLD:
        return 'MEDIUM'
    else:
        return 'HIGH'

test_data_with_predictions = pd.DataFrame({
    'True_Label': y_test.values,
    'Predicted_Probability': y_test_proba,
    'Predicted_Class': y_test_pred,
    'Risk_Level': [classify_risk(p) for p in y_test_proba],
    'Risk_Score': (y_test_proba * 100).astype(int)
})

# Distribution of risk levels
risk_distribution = test_data_with_predictions['Risk_Level'].value_counts()
print(f"\nRisk Level Distribution (Test Set):")
for risk, count in risk_distribution.items():
    pct = (count / len(test_data_with_predictions)) * 100
    print(f"  {risk}: {count} ({pct:.1f}%)")

# Accuracy by risk level
print(f"\nReturn Rate by Risk Level:")
for risk in ['LOW', 'MEDIUM', 'HIGH']:
    subset = test_data_with_predictions[test_data_with_predictions['Risk_Level'] == risk]
    if len(subset) > 0:
        return_rate = subset['True_Label'].mean() * 100
        total = len(subset)
        returned = subset['True_Label'].sum()
        print(f"  {risk}: {returned}/{total} returned ({return_rate:.1f}%)")

# Visualize risk distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Risk level distribution
axes[0].bar(risk_distribution.index, risk_distribution.values, 
           color=['#10B981', '#F59E0B', '#EF4444'])
axes[0].set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Risk Level')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.3)

# Return rate by risk level
return_rates = []
risk_levels = ['LOW', 'MEDIUM', 'HIGH']
colors_map = {'LOW': '#10B981', 'MEDIUM': '#F59E0B', 'HIGH': '#EF4444'}
colors = []

for risk in risk_levels:
    subset = test_data_with_predictions[test_data_with_predictions['Risk_Level'] == risk]
    if len(subset) > 0:
        return_rates.append(subset['True_Label'].mean() * 100)
        colors.append(colors_map[risk])
    else:
        return_rates.append(0)
        colors.append(colors_map[risk])

axes[1].bar(risk_levels, return_rates, color=colors)
axes[1].set_title('Return Rate by Risk Level', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Risk Level')
axes[1].set_ylabel('Return Rate (%)')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim([0, 100])

plt.tight_layout()
plt.savefig('notebooks/14_risk_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notebooks/14_risk_analysis.png")
plt.close()

# ============================================================================
# 11. SAVE TRAINED MODEL
# ============================================================================
print("\n[11] SAVING TRAINED MODEL...")

# Save best XGBoost model
joblib.dump(best_xgb, 'models/xgboost_model.joblib')
print("✓ Saved: models/xgboost_model.joblib")

# Save model metadata
model_metadata = {
    'model_name': 'XGBoost_ReturnFraudDetection',
    'version': '1.0',
    'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_samples': len(X_train),
    'num_features': X_train.shape[1],
    'feature_names': X_train.columns.tolist(),
    'best_params': grid_search.best_params_,
    'performance_metrics': {
        'validation': val_metrics,
        'test': test_metrics
    },
    'risk_thresholds': {
        'low_risk': LOW_RISK_THRESHOLD,
        'high_risk': HIGH_RISK_THRESHOLD
    },
    'class_distribution': {
        'train': {
            'returned': int(y_train.sum()),
            'not_returned': int((y_train == 0).sum()),
            'return_rate': float(y_train.mean())
        },
        'test': {
            'returned': int(y_test.sum()),
            'not_returned': int((y_test == 0).sum()),
            'return_rate': float(y_test.mean())
        }
    }
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=4)
print("✓ Saved: models/model_metadata.json")

# ============================================================================
# 12. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)

print(f"""
MODEL PERFORMANCE:
  Test AUC: {test_metrics['auc']:.4f}
  Test F1-Score: {test_metrics['f1']:.4f}
  Test Accuracy: {test_metrics['accuracy']:.4f}
  Test Precision: {test_metrics['precision']:.4f}
  Test Recall: {test_metrics['recall']:.4f}

IMPROVEMENT OVER BASELINE:
  AUC: +{(test_metrics['auc'] - baseline_metrics['auc'])*100:.2f}%
  F1: +{(test_metrics['f1'] - baseline_metrics['f1'])*100:.2f}%

BEST HYPERPARAMETERS:
  max_depth: {grid_search.best_params_['max_depth']}
  learning_rate: {grid_search.best_params_['learning_rate']}
  n_estimators: {grid_search.best_params_['n_estimators']}
  subsample: {grid_search.best_params_['subsample']}
  colsample_bytree: {grid_search.best_params_['colsample_bytree']}
  min_child_weight: {grid_search.best_params_['min_child_weight']}

RISK THRESHOLDS:
  LOW RISK: probability < {LOW_RISK_THRESHOLD} (Trusted)
  MEDIUM RISK: {LOW_RISK_THRESHOLD} <= probability <= {HIGH_RISK_THRESHOLD} (Uncertain)
  HIGH RISK: probability > {HIGH_RISK_THRESHOLD} (Untrustworthy)

TOP 5 MOST IMPORTANT FEATURES:
""")

for idx, row in feature_importance.head(5).iterrows():
    print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

print(f"""
SAVED ARTIFACTS:
  ✓ models/xgboost_model.joblib
  ✓ models/model_metadata.json
  ✓ models/feature_importance.csv
  ✓ notebooks/10_confusion_matrix.png
  ✓ notebooks/11_roc_curve.png
  ✓ notebooks/12_precision_recall_curve.png
  ✓ notebooks/13_feature_importance.png
  ✓ notebooks/14_risk_analysis.png

NEXT STEPS:
  1. Review visualizations in notebooks/
  2. Verify model performance meets business requirements
  3. Proceed to Phase 3: Python API Service
""")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETED SUCCESSFULLY! ✓")
print("="*80)