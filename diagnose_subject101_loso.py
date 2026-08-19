"""
Diagnose why Subject 101's 'standing' fails under LOSO (train on other 8,
test on 101) despite the saved model (trained WITH 101) getting it perfect.

This script:
  1. Reuses your existing feature-extraction pipeline from
     pamap2_reliability_multisubject.py (same 308 features, same windowing).
  2. Trains the exact same VotingClassifier config, EXCLUDING Subject 101
     from training entirely (true LOSO fold for 101).
  3. Predicts on Subject 101's standing windows.
  4. For the misclassified ones, extracts per-tree feature importances from
     the Random Forest component and shows the top features driving the
     wrong prediction, plus how Subject 101's values for those features
     compare to the TRAINING POPULATION for both the true class (standing)
     and the class it got predicted as (e.g. cycling).

Run this from the same directory as pamap2_combined.csv.
Requires: the same packages already used by pamap2_reliability_multisubject.py
(pandas, numpy, scikit-learn, xgboost, scipy).

This does not modify any of your existing files or cached results.
"""

import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import skew, kurtosis as sp_kurtosis

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

# ── Import your existing feature-extraction code directly ──────────────────
# This avoids duplicating/drifting from your real pipeline's feature logic.
sys.path.insert(0, '.')
try:
    import pamap2_reliability_multisubject as pipeline
except Exception as e:
    print(f"ERROR: could not import pamap2_reliability_multisubject.py: {e}")
    print("Make sure this script is in the same folder as that file.")
    sys.exit(1)

WINDOW_SIZE = pipeline.WINDOW_SIZE
PROTOCOL_ACTIVITY_IDS = pipeline.PROTOCOL_ACTIVITY_IDS
ACTIVITY_MAP = pipeline.ACTIVITY_MAP

TARGET_SUBJECT = 101

print("="*70)
print(f"LOSO DIAGNOSIS — Subject {TARGET_SUBJECT}, true generalization test")
print("="*70)

# ── Step 1: build the full feature dataset using your existing pipeline ────
print("\nBuilding feature dataset (reusing your cached one if present)...")
windows = pipeline.build_feature_dataset()
print(f"Total windows available: {len(windows):,}")

FEATURE_COLS = [c for c in windows.columns if c not in ('activityID', 'subject_id', 'data_source')]
print(f"Feature count: {len(FEATURE_COLS)}")

# ── Step 2: exact same LOSO split logic as pamap2_reliability_multisubject.py
test = windows[
    (windows['subject_id'] == TARGET_SUBJECT) &
    (windows['data_source'] == 'protocol') &
    (windows['activityID'].isin(PROTOCOL_ACTIVITY_IDS))
]
train_prot = windows[
    ~((windows['subject_id'] == TARGET_SUBJECT) & (windows['data_source'] == 'protocol')) &
    (windows['data_source'] == 'protocol') &
    (windows['activityID'].isin(PROTOCOL_ACTIVITY_IDS))
]
train_opt = windows[
    (windows['subject_id'] != TARGET_SUBJECT) &
    (windows['data_source'] == 'optional') &
    (windows['activityID'].isin(PROTOCOL_ACTIVITY_IDS))
]
train = pd.concat([train_prot, train_opt], ignore_index=True)

print(f"\nTrain windows (excludes Subject {TARGET_SUBJECT} entirely): {len(train):,}")
print(f"Test windows (Subject {TARGET_SUBJECT} protocol only):        {len(test):,}")

X_train_raw = train[FEATURE_COLS].values
y_train = train['activityID'].values
X_test_raw = test[FEATURE_COLS].values
y_test = test['activityID'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# ── Step 3: train the exact same VotingClassifier ───────────────────────────
print("\nTraining VotingClassifier (RF + KNN + XGB) excluding Subject 101...")
print("(This will take a few minutes — same config as your real LOSO run)")

rf = RandomForestClassifier(
    n_estimators=300, max_depth=25, min_samples_leaf=2,
    class_weight='balanced', n_jobs=-1, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)
xgb = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    use_label_encoder=False, eval_metric='mlogloss',
    n_jobs=-1, random_state=42, verbosity=0)

clf = VotingClassifier(
    estimators=[('rf', rf), ('knn', knn), ('xgb', xgb)],
    weights=[4, 2, 3], voting='soft', n_jobs=-1)
clf.fit(X_train, y_train)

y_proba = clf.predict_proba(X_test)
classes_ = clf.classes_
y_pred = classes_[y_proba.argmax(axis=1)]
confidence = y_proba.max(axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"\nOverall Subject {TARGET_SUBJECT} LOSO accuracy: {acc:.4f}")

standing_mask = y_test == 3
print(f"\nStanding windows: {standing_mask.sum()}")
print(f"Standing accuracy: {(y_pred[standing_mask] == 3).mean():.4f}")
print("Standing predicted-as distribution:")
print(pd.Series(y_pred[standing_mask]).map(lambda a: ACTIVITY_MAP.get(a, a)).value_counts())

# ── Step 4: feature-level diagnosis on misclassified standing windows ──────
# Use the RF component specifically since it has interpretable feature_importances_
rf_fitted = clf.named_estimators_['rf']
importances = rf_fitted.feature_importances_
top_idx = np.argsort(importances)[::-1][:20]
top_feats = [FEATURE_COLS[i] for i in top_idx]

print(f"\n{'='*70}")
print("TOP 20 RF FEATURE IMPORTANCES (trained without Subject 101)")
print(f"{'='*70}")
for rank, i in enumerate(top_idx, 1):
    print(f"  {rank:>2}. {FEATURE_COLS[i]:<35} importance={importances[i]:.4f}")

# For misclassified standing windows, compare Subject 101's values on these
# top features against: (a) training population's standing class, and
# (b) training population's class it was WRONGLY predicted as.
wrong_standing_mask = standing_mask & (y_pred != 3)
n_wrong = wrong_standing_mask.sum()
print(f"\n{'='*70}")
print(f"MISCLASSIFIED STANDING WINDOWS: {n_wrong} / {standing_mask.sum()}")
print(f"{'='*70}")

if n_wrong > 0:
    wrong_preds = y_pred[wrong_standing_mask]
    most_common_wrong = pd.Series(wrong_preds).mode()[0]
    most_common_wrong_name = ACTIVITY_MAP.get(most_common_wrong, most_common_wrong)
    print(f"\nMost common wrong prediction: {most_common_wrong_name} (class {most_common_wrong})")

    X_test_wrong_unscaled = X_test_raw[wrong_standing_mask]

    train_standing_mask = y_train == 3
    train_wrongclass_mask = y_train == most_common_wrong

    print(f"\n{'feature':<32} {'S101 wrong-standing':>20} {'train standing avg':>20} {'train ' + str(most_common_wrong_name) + ' avg':>20}")
    for feat in top_feats[:15]:
        idx = FEATURE_COLS.index(feat)
        s101_val = X_test_wrong_unscaled[:, idx].mean()
        train_standing_val = X_train_raw[train_standing_mask, idx].mean()
        train_wrongclass_val = X_train_raw[train_wrongclass_mask, idx].mean()
        # Which is S101 closer to?
        dist_to_standing = abs(s101_val - train_standing_val)
        dist_to_wrongclass = abs(s101_val - train_wrongclass_val)
        closer = "→ WRONG CLASS" if dist_to_wrongclass < dist_to_standing else "→ standing"
        print(f"{feat:<32} {s101_val:>20.4f} {train_standing_val:>20.4f} {train_wrongclass_val:>20.4f}  {closer}")

print(f"\n{'='*70}")
print("DONE — this shows exactly which features pull Subject 101's standing")
print("windows toward the wrong class, in the true LOSO (101 excluded) setting.")
print(f"{'='*70}")
