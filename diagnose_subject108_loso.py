"""
Diagnose Subject 108's ironing failure (40.4% vs 94-99.8% for all other
subjects), using the same LOSO methodology already validated for Subject 101.

Reuses pamap2_reliability_multisubject.py's feature pipeline exactly.
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

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

sys.path.insert(0, '.')
import pamap2_reliability_multisubject as pipeline

WINDOW_SIZE = pipeline.WINDOW_SIZE
PROTOCOL_ACTIVITY_IDS = pipeline.PROTOCOL_ACTIVITY_IDS
ACTIVITY_MAP = pipeline.ACTIVITY_MAP

TARGET_SUBJECT = 108

print("="*70)
print(f"LOSO DIAGNOSIS — Subject {TARGET_SUBJECT}, ironing failure")
print("="*70)

windows = pipeline.build_feature_dataset()
FEATURE_COLS = [c for c in windows.columns if c not in ('activityID', 'subject_id', 'data_source')]

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

print("\nTraining VotingClassifier excluding Subject 108...")
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

acc = accuracy_score(y_test, y_pred)
print(f"\nOverall Subject {TARGET_SUBJECT} LOSO accuracy: {acc:.4f}")

ironing_mask = y_test == 17
print(f"\nIroning windows: {ironing_mask.sum()}")
print(f"Ironing accuracy: {(y_pred[ironing_mask] == 17).mean():.4f}")
print("Ironing predicted-as distribution:")
print(pd.Series(y_pred[ironing_mask]).map(lambda a: ACTIVITY_MAP.get(a, a)).value_counts())

rf_fitted = clf.named_estimators_['rf']
importances = rf_fitted.feature_importances_
top_idx = np.argsort(importances)[::-1][:20]
top_feats = [FEATURE_COLS[i] for i in top_idx]

print(f"\n{'='*70}")
print("TOP 20 RF FEATURE IMPORTANCES (trained without Subject 108)")
print(f"{'='*70}")
for rank, i in enumerate(top_idx, 1):
    print(f"  {rank:>2}. {FEATURE_COLS[i]:<35} importance={importances[i]:.4f}")

wrong_ironing_mask = ironing_mask & (y_pred != 17)
n_wrong = wrong_ironing_mask.sum()
print(f"\n{'='*70}")
print(f"MISCLASSIFIED IRONING WINDOWS: {n_wrong} / {ironing_mask.sum()}")
print(f"{'='*70}")

if n_wrong > 0:
    wrong_preds = y_pred[wrong_ironing_mask]
    most_common_wrong = pd.Series(wrong_preds).mode()[0]
    most_common_wrong_name = ACTIVITY_MAP.get(most_common_wrong, most_common_wrong)
    print(f"\nMost common wrong prediction: {most_common_wrong_name} (class {most_common_wrong})")

    X_test_wrong_unscaled = X_test_raw[wrong_ironing_mask]
    train_ironing_mask = y_train == 17
    train_wrongclass_mask = y_train == most_common_wrong

    print(f"\n{'feature':<32} {'S108 wrong-ironing':>20} {'train ironing avg':>20} {'train ' + str(most_common_wrong_name) + ' avg':>20}")
    for feat in top_feats[:15]:
        idx = FEATURE_COLS.index(feat)
        s108_val = X_test_wrong_unscaled[:, idx].mean()
        train_ironing_val = X_train_raw[train_ironing_mask, idx].mean()
        train_wrongclass_val = X_train_raw[train_wrongclass_mask, idx].mean()
        dist_to_ironing = abs(s108_val - train_ironing_val)
        dist_to_wrongclass = abs(s108_val - train_wrongclass_val)
        closer = "-> WRONG CLASS" if dist_to_wrongclass < dist_to_ironing else "-> ironing"
        print(f"{feat:<32} {s108_val:>20.4f} {train_ironing_val:>20.4f} {train_wrongclass_val:>20.4f}  {closer}")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
