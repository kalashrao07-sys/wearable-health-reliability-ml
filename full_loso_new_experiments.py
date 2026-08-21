"""
Full 8-subject LOSO for the two experiments promoted from single-subject
screening:
  1. Stacked ensemble (LightGBM + ExtraTrees + SVM base learners ->
     LogisticRegression meta-learner) — same base-learner choice as the
     single-subject screening run that showed a promising signal.
  2. SVM-RBF, standalone — no existing data at any protocol, cleanest test
     of a different decision-boundary geometry.

Both use IDENTICAL conditions to the validated baseline:
  - Same 308-feature set (pamap2_reliability_multisubject.py pipeline)
  - Same 8-subject Protocol-only LOSO (subject fully excluded from training
    each fold, scaler fit only on training fold)
  - Same GREEN/YELLOW/RED thresholds (0.75, 0.45)
  - Same reliability metrics: accuracy, balanced accuracy, macro F1, ECE,
    per-activity accuracy for stairs_up/down and ironing/vacuuming

QML excluded (near-random single-subject result — investigate implementation
separately before spending compute on a full LOSO run).
TCN excluded from this round (ambiguous single-subject signal, deferred).
Transformer deferred as explicit "if time permits" — not run in this script.
"""

import sys
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score)
from sklearn.base import clone

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed. Run: pip install lightgbm")
    sys.exit(1)

sys.path.insert(0, '.')
import pamap2_reliability_multisubject as pipeline

PROTOCOL_ACTIVITY_IDS = pipeline.PROTOCOL_ACTIVITY_IDS
ACTIVITY_MAP = pipeline.ACTIVITY_MAP

ALL_SUBJECTS_TO_TEST = [101, 102, 103, 104, 105, 106, 107, 108]
STAIRS_ACTIVITIES = {12: 'stairs_up', 13: 'stairs_down'}
IRONING_VACUUM = {17: 'ironing', 16: 'vacuuming'}


def traffic_light(conf):
    if conf >= 0.75:
        return "GREEN"
    elif conf >= 0.45:
        return "YELLOW"
    return "RED"


def expected_calibration_error(confidences, corrects, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = corrects[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return ece


def build_stacked_ensemble():
    """
    Same base-learner choice as pamap2_stacked_ensemble.py's single-subject
    run that showed a promising signal (93.2% acc, strong on target pairs).
    Wrapped as a StackingClassifier for identical LOSO harness reuse.
    """
    from sklearn.ensemble import StackingClassifier
    lgbm = lgb.LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        num_leaves=31, class_weight='balanced',
        n_jobs=-1, random_state=42, verbosity=-1)
    extra_trees = ExtraTreesClassifier(
        n_estimators=300, max_depth=25, min_samples_leaf=2,
        class_weight='balanced', n_jobs=-1, random_state=42)
    svm = SVC(kernel='rbf', C=10.0, gamma='scale',
              probability=True, class_weight='balanced', random_state=42)

    return StackingClassifier(
        estimators=[('lgbm', lgbm), ('extra_trees', extra_trees), ('svm', svm)],
        final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced'),
        cv=3, n_jobs=-1, passthrough=False)


def build_svm_rbf():
    return SVC(kernel='rbf', C=10.0, gamma='scale',
               probability=True, class_weight='balanced', random_state=42)


def run_loso_for_model(model_name, model_template, windows, feature_cols):
    all_results = []

    for target_subject in ALL_SUBJECTS_TO_TEST:
        test = windows[
            (windows['subject_id'] == target_subject) &
            (windows['data_source'] == 'protocol') &
            (windows['activityID'].isin(PROTOCOL_ACTIVITY_IDS))
        ]
        if len(test) == 0:
            continue

        train_prot = windows[
            ~((windows['subject_id'] == target_subject) & (windows['data_source'] == 'protocol')) &
            (windows['data_source'] == 'protocol') &
            (windows['activityID'].isin(PROTOCOL_ACTIVITY_IDS))
        ]
        train_opt = windows[
            (windows['subject_id'] != target_subject) &
            (windows['data_source'] == 'optional') &
            (windows['activityID'].isin(PROTOCOL_ACTIVITY_IDS))
        ]
        train = pd.concat([train_prot, train_opt], ignore_index=True)

        X_train_raw = train[feature_cols].values
        y_train = train['activityID'].values
        X_test_raw = test[feature_cols].values
        y_test = test['activityID'].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model = clone(model_template)

        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0

        y_proba = model.predict_proba(X_test)
        classes_ = model.classes_
        y_pred = classes_[y_proba.argmax(axis=1)]
        confidence = y_proba.max(axis=1)

        acc = accuracy_score(y_test, y_pred)
        print(f"    Subject {target_subject}: acc={acc:.4f}  (fit={fit_time:.1f}s, n_test={len(y_test)})")

        for i in range(len(y_test)):
            all_results.append({
                'model': model_name,
                'subject': target_subject,
                'actual_activity': ACTIVITY_MAP.get(int(y_test[i]), str(y_test[i])),
                'predicted_activity': ACTIVITY_MAP.get(int(y_pred[i]), str(y_pred[i])),
                'actual_id': int(y_test[i]),
                'predicted_id': int(y_pred[i]),
                'confidence': float(confidence[i]),
                'is_correct': int(y_test[i] == y_pred[i]),
            })

    return pd.DataFrame(all_results)


def summarize_model(df, model_name):
    print(f"\n{'='*70}")
    print(f"SUMMARY — {model_name}  (8-subject LOSO, n={len(df)})")
    print(f"{'='*70}")

    y_true = df['actual_id'].values
    y_pred = df['predicted_id'].values
    conf = df['confidence'].values
    correct = df['is_correct'].values

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    ece = expected_calibration_error(conf, correct)

    print(f"  Overall accuracy:     {acc:.4f}")
    print(f"  Balanced accuracy:    {bal_acc:.4f}")
    print(f"  Macro F1:             {macro_f1:.4f}")
    print(f"  Macro precision:      {macro_prec:.4f}")
    print(f"  Macro recall:         {macro_rec:.4f}")
    print(f"  Mean confidence:      {conf.mean():.4f}")
    print(f"  ECE:                  {ece:.4f}")

    df = df.copy()
    df['traffic_light'] = df['confidence'].apply(traffic_light)
    print(f"\n  Reliability tiers:")
    for tier in ['GREEN', 'YELLOW', 'RED']:
        sub = df[df['traffic_light'] == tier]
        if len(sub) == 0:
            continue
        print(f"    {tier:<8} n={len(sub):>6}  ({len(sub)/len(df):.1%})  "
              f"accuracy={sub['is_correct'].mean():.4f}")

    print(f"\n  Per-activity accuracy (target confusion pairs):")
    for aid, aname in {**STAIRS_ACTIVITIES, **IRONING_VACUUM}.items():
        sub = df[df['actual_id'] == aid]
        if len(sub) == 0:
            continue
        print(f"    {aname:<15} n={len(sub):>5}  accuracy={sub['is_correct'].mean():.4f}")

    print(f"\n  Per-subject accuracy (check for idiosyncratic subject failures,"
          f" same diagnostic used for Subject 101/108):")
    per_subj = df.groupby('subject')['is_correct'].agg(['mean', 'count'])
    print(per_subj.round(4))

    return {
        'model': model_name, 'accuracy': acc, 'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1, 'ece': ece, 'mean_confidence': conf.mean(),
    }


# ── MAIN ─────────────────────────────────────────────────────────────────
print("="*70)
print("FULL 8-SUBJECT LOSO — NEW EXPERIMENTS")
print("(Stacked ensemble + SVM-RBF, promoted from single-subject screening)")
print("="*70)

print("\nLoading feature dataset (reuses existing cache if present)...")
windows = pipeline.build_feature_dataset()
feature_cols = [c for c in windows.columns if c not in ('activityID', 'subject_id', 'data_source')]
print(f"Feature count: {len(feature_cols)}")
assert len(feature_cols) == 308, f"Expected 308 features, got {len(feature_cols)} — check cache/pipeline."

experiments = {
    'stacked_ensemble_lgbm_et_svm': build_stacked_ensemble(),
    'svm_rbf_standalone': build_svm_rbf(),
}

summary_rows = []

for model_name, model_template in experiments.items():
    print(f"\n{'='*70}")
    print(f"RUNNING LOSO — {model_name}")
    print(f"{'='*70}")
    df_result = run_loso_for_model(model_name, model_template, windows, feature_cols)
    df_result.to_csv(f'pamap2_loso_{model_name}.csv', index=False)
    summary = summarize_model(df_result, model_name)
    summary_rows.append(summary)

# ── LOAD EXISTING BASELINE FOR SIDE-BY-SIDE COMPARISON ──────────────────
print(f"\n{'='*70}")
print("LOADING EXISTING BASELINE (already-validated 8-subject LOSO result)")
print(f"{'='*70}")
try:
    baseline_df = pd.read_csv('pamap2_multisubject_reliability_final.csv')
    baseline_df = baseline_df.rename(columns={
        'actual_activity': 'actual_activity', 'predicted_activity': 'predicted_activity'
    })
    y_true_b = baseline_df['actual_activity'].map(
        {v: k for k, v in ACTIVITY_MAP.items()}).values
    y_pred_b = baseline_df['predicted_activity'].map(
        {v: k for k, v in ACTIVITY_MAP.items()}).values
    conf_b = baseline_df['confidence'].values
    correct_b = baseline_df['is_correct'].values

    acc_b = accuracy_score(y_true_b, y_pred_b)
    bal_acc_b = balanced_accuracy_score(y_true_b, y_pred_b)
    macro_f1_b = f1_score(y_true_b, y_pred_b, average='macro')
    ece_b = expected_calibration_error(conf_b, correct_b)

    summary_rows.append({
        'model': 'baseline_voting_rf_knn_xgb (existing)',
        'accuracy': acc_b, 'balanced_accuracy': bal_acc_b,
        'macro_f1': macro_f1_b, 'ece': ece_b, 'mean_confidence': conf_b.mean(),
    })
    print("Baseline loaded successfully and included in comparison.")
except FileNotFoundError:
    print("WARNING: pamap2_multisubject_reliability_final.csv not found in this "
          "directory — baseline NOT included in final table. Copy it here to "
          "get a complete comparison.")

# ── FINAL COMPARISON TABLE ──────────────────────────────────────────────
print(f"\n{'='*70}")
print("FINAL COMPARISON — BASELINE + NEW EXPERIMENTS (all 8-subject LOSO)")
print(f"{'='*70}")
summary_df = pd.DataFrame(summary_rows).set_index('model')
print(summary_df.round(4))
summary_df.to_csv('pamap2_new_experiments_comparison_summary.csv')

print(f"\n{'='*70}")
print("DONE")
print("Per-window results: pamap2_loso_<model>.csv")
print("Summary: pamap2_new_experiments_comparison_summary.csv")
print(f"{'='*70}")
