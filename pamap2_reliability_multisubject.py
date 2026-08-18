"""
PAMAP2 — Multi-Subject Reliability Validation (FINAL PIPELINE)

Validates whether the traffic-light confidence system (GREEN >= 0.75,
YELLOW 0.45-0.75, RED < 0.45) actually corresponds to different levels
of prediction reliability — using the SAME 308-feature engineering and
the SAME final VotingClassifier (RF + KNN + XGBoost, weights [4,2,3])
as pamap2_ml_model.py. No standalone/simplified model is used here.

For each validation subject:
    Train  = all other subjects (Protocol + Optional, minus that subject's Protocol data)
    Test   = that subject's Protocol-only windows (unseen, never touched by scaler/model)

This mirrors the LOSO principle used in the main pipeline's TEST_SUBJECT split,
just repeated across several subjects instead of one.

Run AFTER pamap2_ml_model.py (needs pamap2_combined.csv in the same directory).

IMPORTANT: thresholds, feature set, and model config are NOT tuned here.
This script only measures whether confidence predicts correctness.

── INCREMENTAL / RESUMABLE WORKFLOW ────────────────────────────────────────
Subjects 103, 105, 106, 108 have already been validated and their per-window
results live in pamap2_multisubject_reliability.csv. This script:
  1. Caches the complete 308-feature dataset (pamap2_reliability_features.pkl,
     a pickled DataFrame with all feature columns + activityID + subject_id +
     data_source) so feature extraction from the raw CSV and the inter-window
     stability-feature step never have to run twice.
  2. Only trains/validates the subjects listed in VALIDATE_SUBJECTS below —
     currently the five remaining ones. It does NOT rerun 103/105/106/108.
  3. Saves the new subjects' per-window results to a SEPARATE file
     (pamap2_multisubject_reliability_remaining.csv) so the original
     4-subject file is never overwritten.
  4. A separate combine stage (bottom of this script) merges the existing
     4-subject file with the new 5-subject file into the final 9-subject
     per-window dataset, verifies all 9 subjects are present, and recomputes
     every statistic directly from that combined per-window data — never by
     averaging the old and new summary numbers.
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              precision_score, recall_score, f1_score)
from scipy.stats import skew, kurtosis as sp_kurtosis, spearmanr

try:
    from xgboost import XGBClassifier
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CONFIG (must match pamap2_ml_model.py exactly) ─────────────────────────
INPUT_FILE        = "pamap2_combined.csv"
WINDOW_SIZE        = 100
STEP_SIZE          = 50
SAMPLING_FREQ      = 100
PURITY_THRESHOLD   = 0.85
SMOOTH_WINDOW       = 29     # not used for reliability windows (per-window eval), kept for parity

FEATURE_CACHE_PKL  = "pamap2_reliability_features.pkl"

EXISTING_RESULTS_CSV  = "pamap2_multisubject_reliability.csv"            # 103, 105, 106, 108 — DO NOT OVERWRITE
REMAINING_RESULTS_CSV = "pamap2_multisubject_reliability_remaining.csv"  # new 5-subject run
FINAL_RESULTS_CSV     = "pamap2_multisubject_reliability_final.csv"      # combined 9-subject per-window data

ALREADY_VALIDATED_SUBJECTS = [103, 105, 106, 108]
ALL_NINE_SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 108, 109]

# Only the five remaining subjects are trained/validated in this run.
# 103, 105, 106, 108 are NOT rerun — their results already exist in
# pamap2_multisubject_reliability.csv and are picked up by the combine stage.
VALIDATE_SUBJECTS = [101, 102, 104, 107, 109]

ACTIVITY_MAP = {
    1:'lying', 2:'sitting', 3:'standing', 4:'walking', 5:'running',
    6:'cycling', 7:'nordic_walk', 9:'watching_TV', 10:'computer_work',
    11:'car_driving', 12:'stairs_up', 13:'stairs_down', 16:'vacuuming',
    17:'ironing', 18:'folding_laundry', 19:'house_clean', 20:'soccer', 24:'rope_jumping'
}
PROTOCOL_ACTIVITY_IDS = {1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17}

print("=" * 65)
print("PAMAP2 — MULTI-SUBJECT RELIABILITY VALIDATION (FINAL MODEL)")
print("=" * 65)

# ── VERIFY EXISTING 4-SUBJECT RESULTS BEFORE DOING ANYTHING ELSE ──────────
# Required by the resumable workflow: load the already-completed results,
# confirm they contain exactly 103/105/106/108, and stop rather than
# silently proceeding toward an incomplete final result if not.
if os.path.exists(EXISTING_RESULTS_CSV):
    _existing_check_df = pd.read_csv(EXISTING_RESULTS_CSV)
    _existing_subjects = set(_existing_check_df['subject'].unique().tolist())
    _expected_existing = set(ALREADY_VALIDATED_SUBJECTS)
    if _existing_subjects != _expected_existing:
        print(f"WARNING: {EXISTING_RESULTS_CSV} does not contain exactly "
              f"{sorted(_expected_existing)}.")
        print(f"  Found subjects: {sorted(_existing_subjects)}")
        raise SystemExit(
            "Stopping — existing 4-subject results are missing or unexpected. "
            "Refusing to proceed toward an incomplete final result."
        )
    print("Existing validation results found:")
    print(f"  {', '.join(str(s) for s in sorted(_existing_subjects))}")
else:
    print(f"WARNING: {EXISTING_RESULTS_CSV} not found.")
    raise SystemExit(
        f"Stopping — {EXISTING_RESULTS_CSV} (results for "
        f"{ALREADY_VALIDATED_SUBJECTS}) must exist before running the "
        "remaining subjects."
    )

print("\nRunning remaining subjects:")
for _s in VALIDATE_SUBJECTS:
    print(f"  {_s}")
print("=" * 65)


# ── FEATURE ENGINEERING — identical to pamap2_ml_model.py ─────────────────
def spectral_entropy(signal):
    fft_mag = np.abs(np.fft.rfft(signal))
    power = fft_mag ** 2
    total = power.sum()
    if total == 0:
        return 0.0
    prob = power / total
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def safe_corr(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_window_features(data: pd.DataFrame) -> pd.DataFrame:
    """Identical feature set/order to pamap2_ml_model.py — 308 features total."""
    freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / SAMPLING_FREQ)
    records = []
    discarded_impure = 0

    for start in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        window = data.iloc[start: start + WINDOW_SIZE]
        label = window['activityID'].mode()[0]
        purity = (window['activityID'] == label).mean()
        subject = window['subject_id'].iloc[0]

        if label == 24:
            continue
        if purity < PURITY_THRESHOLD:
            discarded_impure += 1
            continue

        feat = {}

        for col in SENSOR_COLS:
            vals = window[col].values
            feat[f"{col}_mean"] = vals.mean()
            feat[f"{col}_std"] = vals.std()
            feat[f"{col}_min"] = vals.min()
            feat[f"{col}_max"] = vals.max()
            feat[f"{col}_range"] = vals.max() - vals.min()
            feat[f"{col}_energy"] = (vals ** 2).mean()
            fft_mag = np.abs(np.fft.rfft(vals - vals.mean()))
            power = fft_mag ** 2
            dom_idx = np.argmax(power)
            feat[f"{col}_dom_freq"] = freqs[dom_idx]
            feat[f"{col}_peak_power"] = power[dom_idx]
            feat[f"{col}_spectral_entropy"] = spectral_entropy(vals)

        mags = {}
        for loc in ['hand', 'chest', 'ankle']:
            x = window[f'{loc}_acc16_x'].values
            y = window[f'{loc}_acc16_y'].values
            z = window[f'{loc}_acc16_z'].values
            mag = np.sqrt(x**2 + y**2 + z**2)
            mags[loc] = mag
            feat[f'{loc}_acc_mag_mean'] = mag.mean()
            feat[f'{loc}_acc_mag_std'] = mag.std()
            delta = np.diff(mag)
            feat[f'{loc}_acc_mag_delta_mean'] = np.abs(delta).mean()
            feat[f'{loc}_acc_mag_delta_std'] = delta.std()

        gyro_mags = {}
        for loc in ['hand', 'chest', 'ankle']:
            gx = window[f'{loc}_gyro_x'].values
            gy = window[f'{loc}_gyro_y'].values
            gz = window[f'{loc}_gyro_z'].values
            gm = np.sqrt(gx**2 + gy**2 + gz**2)
            gyro_mags[loc] = gm
            feat[f'{loc}_gyro_mag_mean'] = gm.mean()
            feat[f'{loc}_gyro_mag_std'] = gm.std()
            feat[f'{loc}_gyro_mag_max'] = gm.max()

        for ax in ['x', 'y', 'z']:
            vals_centered = window[f'hand_acc16_{ax}'].values
            vals_centered = vals_centered - vals_centered.mean()
            zcr = float(((vals_centered[:-1] * vals_centered[1:]) < 0).sum()) / len(vals_centered)
            feat[f'hand_acc16_{ax}_zcr'] = zcr
        for ax in ['x', 'z']:
            vals_centered = window[f'ankle_acc16_{ax}'].values
            vals_centered = vals_centered - vals_centered.mean()
            zcr = float(((vals_centered[:-1] * vals_centered[1:]) < 0).sum()) / len(vals_centered)
            feat[f'ankle_acc16_{ax}_zcr'] = zcr

        for lag in [10, 25, 50]:
            m_hand = mags['hand']
            m_centered = m_hand - m_hand.mean()
            if m_centered.std() > 1e-8:
                ac = float(np.corrcoef(m_centered[:-lag], m_centered[lag:])[0, 1])
                feat[f'hand_acc_mag_autocorr_{lag}'] = ac if np.isfinite(ac) else 0.0
            else:
                feat[f'hand_acc_mag_autocorr_{lag}'] = 0.0

        for col in ['hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z',
                    'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z',
                    'chest_gyro_x', 'chest_gyro_y']:
            vals = window[col].values
            feat[f'{col}_skew'] = float(skew(vals))
            feat[f'{col}_kurt'] = float(sp_kurtosis(vals))

        body_motion = gyro_mags['chest'].std() + gyro_mags['ankle'].std()
        feat['hand_gyro_dominance'] = gyro_mags['hand'].std() / (body_motion + 1e-6)

        feat['corr_hand_ankle'] = safe_corr(mags['hand'], mags['ankle'])
        feat['corr_hand_chest'] = safe_corr(mags['hand'], mags['chest'])
        feat['corr_chest_ankle'] = safe_corr(mags['chest'], mags['ankle'])

        hand_std = mags['hand'].std()
        ankle_std = mags['ankle'].std()
        chest_std = mags['chest'].std()
        feat['hand_ankle_ratio'] = hand_std / (ankle_std + 1e-6)
        feat['hand_chest_ratio'] = hand_std / (chest_std + 1e-6)
        feat['chest_ankle_ratio'] = chest_std / (ankle_std + 1e-6)

        for loc in ['hand', 'chest', 'ankle']:
            feat[f'{loc}_tilt_proxy'] = window[f'{loc}_acc16_z'].values.mean()

        feat['activityID'] = label
        feat['subject_id'] = subject
        feat['data_source'] = window['data_source'].iloc[0] if 'data_source' in window.columns else 'protocol'
        records.append(feat)

    if discarded_impure > 0:
        print(f"    Discarded {discarded_impure:,} impure windows (purity < {PURITY_THRESHOLD:.0%})")
    return pd.DataFrame(records)


def build_feature_dataset():
    """Runs the full 308-feature extraction over the raw CSV. Only called
    when no cache is present."""
    print("\nLoading combined dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Shape: {df.shape}")

    global SENSOR_COLS
    SENSOR_COLS = [c for c in df.columns
                   if c not in ['timestamp', 'activityID', 'subject_id',
                                 'activity_name', 'acc_magnitude', 'data_source',
                                 'heartRate',
                                 'hand_temp', 'chest_temp', 'ankle_temp']]
    print(f"  Sensor columns used: {len(SENSOR_COLS)}")

    print("\nExtracting features for all subjects (same 308-feature pipeline)...")
    all_windows = []
    for sid in df['subject_id'].unique():
        print(f"  Processing subject {sid}...")
        subj_df = df[df['subject_id'] == sid].reset_index(drop=True)
        all_windows.append(extract_window_features(subj_df))

    windows_ = pd.concat(all_windows, ignore_index=True).dropna()

    # ── INTER-WINDOW STABILITY FEATURES — identical to main pipeline ──────
    STABILITY_FEATURES = [
        'hand_acc_mag_mean', 'hand_acc_mag_std',
        'hand_acc16_x_mean', 'hand_acc16_y_mean', 'hand_acc16_z_mean',
        'hand_ankle_ratio',
        'hand_gyro_mag_mean', 'hand_gyro_mag_std',
        'hand_acc_mag_autocorr_25',
        'hand_gyro_dominance',
    ]

    stability_dfs = []
    for sid in windows_['subject_id'].unique():
        subj = windows_[windows_['subject_id'] == sid].copy()
        for feat in STABILITY_FEATURES:
            if feat in subj.columns:
                subj[f'{feat}_rolling_std'] = (
                    subj[feat].rolling(window=5, center=True, min_periods=1).std().fillna(0)
                )
        stability_dfs.append(subj)

    windows_ = pd.concat(stability_dfs, ignore_index=True)
    return windows_


SENSOR_COLS = None  # populated inside build_feature_dataset() on a cache miss

if os.path.exists(FEATURE_CACHE_PKL):
    print("Loading cached 308-feature dataset...")
    windows = pd.read_pickle(FEATURE_CACHE_PKL)
    FEATURE_COLS = [c for c in windows.columns if c not in ['activityID', 'subject_id', 'data_source']]
    print(f"  Loaded {len(windows):,} windows | {len(FEATURE_COLS)} features from cache")
else:
    print("Feature cache not found. Extracting features...")
    windows = build_feature_dataset()

    FEATURE_COLS = [c for c in windows.columns if c not in ['activityID', 'subject_id', 'data_source']]
    assert len(FEATURE_COLS) == 308, (
        f"Expected 308 features to match the final pipeline, got {len(FEATURE_COLS)}. "
        "Check that this script's feature engineering hasn't drifted from pamap2_ml_model.py."
    )

    print(f"\nSaving feature cache ({len(windows):,} windows x {len(FEATURE_COLS)} features)...")
    windows.to_pickle(FEATURE_CACHE_PKL)
    print(f"  Saved: {FEATURE_CACHE_PKL}")

print(f"\nTotal windows: {len(windows):,} | Features: {len(FEATURE_COLS)}")
assert len(FEATURE_COLS) == 308, (
    f"Expected 308 features to match the final pipeline, got {len(FEATURE_COLS)}. "
    "Cache may be stale — delete pamap2_reliability_features.pkl and rerun."
)


def traffic_light(c):
    if c >= 0.75:
        return "GREEN"
    if c >= 0.45:
        return "YELLOW"
    return "RED"


def build_voting_clf():
    """Exact same VotingClassifier config as pamap2_ml_model.py."""
    return VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=300, max_depth=25, min_samples_leaf=2,
                class_weight='balanced', n_jobs=-1, random_state=42)),
            ('knn', KNeighborsClassifier(
                n_neighbors=7, weights='distance', n_jobs=-1)),
            ('xgb', XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                use_label_encoder=False, eval_metric='mlogloss',
                n_jobs=-1, random_state=42, verbosity=0)),
        ],
        weights=[4, 2, 3],
        voting='soft',
        n_jobs=-1
    )


# ── RUN SUBJECT-WISE VALIDATION WITH THE FINAL VOTING CLASSIFIER ──────────
# Skip any subject already validated in pamap2_multisubject_reliability.csv —
# defensive guard against accidentally rerunning 103/105/106/108.
subjects_to_run = [s for s in VALIDATE_SUBJECTS if s not in ALREADY_VALIDATED_SUBJECTS]
skipped_already_done = [s for s in VALIDATE_SUBJECTS if s in ALREADY_VALIDATED_SUBJECTS]
if skipped_already_done:
    print(f"\nSkipping subjects already validated (found in {EXISTING_RESULTS_CSV}): {skipped_already_done}")

all_results = []
per_subject_summary = []

for val_sub in subjects_to_run:
    print(f"\n{'='*65}")
    print(f"Validating on Subject {val_sub}")
    print(f"{'='*65}")

    test = windows[
        (windows['subject_id'] == val_sub) &
        (windows['data_source'] == 'protocol') &
        (windows['activityID'].isin(PROTOCOL_ACTIVITY_IDS))
    ]
    if len(test) == 0:
        print(f"  Skipped — no Protocol data for subject {val_sub}")
        continue

    train_prot = windows[
        ~((windows['subject_id'] == val_sub) & (windows['data_source'] == 'protocol')) &
        (windows['data_source'] == 'protocol') &
        (windows['activityID'].isin(PROTOCOL_ACTIVITY_IDS))
    ]
    train_opt = windows[
        (windows['subject_id'] != val_sub) &
        (windows['data_source'] == 'optional') &
        (windows['activityID'].isin(PROTOCOL_ACTIVITY_IDS))
    ]
    train = pd.concat([train_prot, train_opt], ignore_index=True)

    X_train_raw = train[FEATURE_COLS].values
    y_train = train['activityID'].values
    X_test_raw = test[FEATURE_COLS].values
    y_test = test['activityID'].values

    # Scaler fit ONLY on training subjects — never on the validation subject.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    clf = build_voting_clf()
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)
    classes_ = clf.classes_
    y_pred = classes_[y_proba.argmax(axis=1)]
    confidence = y_proba.max(axis=1)

    acc = accuracy_score(y_test, y_pred)
    bac = balanced_accuracy_score(y_test, y_pred)
    is_correct = (y_pred == y_test).astype(int)
    tls = np.array([traffic_light(c) for c in confidence])

    print(f"  Accuracy: {acc:.1%}  |  Balanced Accuracy: {bac:.1%}  |  Windows: {len(y_test):,}")

    subj_row = {
        "subject": val_sub,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bac, 4),
        "n_windows": len(y_test),
    }
    for tl in ['GREEN', 'YELLOW', 'RED']:
        mask = tls == tl
        n = mask.sum()
        tier_acc = is_correct[mask].mean() if n > 0 else np.nan
        print(f"    {tl:<8} {n:,} windows  |  Accuracy: {tier_acc:.1%}" if n > 0
              else f"    {tl:<8} 0 windows")
        subj_row[f"{tl.lower()}_accuracy"] = round(tier_acc, 4) if n > 0 else None
        subj_row[f"{tl.lower()}_n"] = int(n)
    per_subject_summary.append(subj_row)

    df_result = pd.DataFrame({
        'window_id': [f"{val_sub}_{i}" for i in range(len(y_test))],
        'subject': val_sub,
        'confidence': confidence.round(4),
        'predicted_activity': [ACTIVITY_MAP.get(a, str(a)) for a in y_pred],
        'actual_activity': [ACTIVITY_MAP.get(a, str(a)) for a in y_test],
        'is_correct': is_correct,
        'traffic_light': tls,
    })
    all_results.append(df_result)

# ── SAVE NEW SUBJECTS' RESULTS SEPARATELY — never touches the existing file ─
if all_results:
    new_results = pd.concat(all_results, ignore_index=True)
    new_results.to_csv(REMAINING_RESULTS_CSV, index=False)
    print(f"\nSaved new per-window results: {REMAINING_RESULTS_CSV} ({len(new_results):,} windows)")

    per_subject_new_df = pd.DataFrame(per_subject_summary)
    per_subject_new_df.to_csv("pamap2_per_subject_summary_remaining.csv", index=False)
    print(f"Saved: pamap2_per_subject_summary_remaining.csv")
else:
    print("\nNo new subjects were validated this run (nothing in subjects_to_run).")


# ════════════════════════════════════════════════════════════════════════
# COMBINE STAGE — merge existing 4-subject results + new 5-subject results,
# verify all 9 subjects present, and recompute every statistic from the
# combined per-window data. Nothing here is averaged from old summaries.
# ════════════════════════════════════════════════════════════════════════

def traffic_light_from_conf(c):
    if c >= 0.75:
        return "GREEN"
    if c >= 0.45:
        return "YELLOW"
    return "RED"


print(f"\n{'='*65}")
print("COMBINE STAGE — building final 9-subject dataset")
print(f"{'='*65}")

pieces = []
if os.path.exists(EXISTING_RESULTS_CSV):
    existing_df = pd.read_csv(EXISTING_RESULTS_CSV)

    # Note whether the existing 4-subject file has activity-label columns.
    # These are only needed for macro Precision/Recall/F1 — their absence
    # does NOT stop the run. The existing results are preserved exactly as
    # they are; missing-label handling happens later, at the metrics stage.
    required_label_cols = {'predicted_activity', 'actual_activity'}
    missing_label_cols = required_label_cols - set(existing_df.columns)
    if missing_label_cols:
        print(f"  Note: {EXISTING_RESULTS_CSV} is missing column(s) "
              f"{sorted(missing_label_cols)}. Macro Precision/Recall/F1 will be "
              "omitted from the final metrics (all other statistics are unaffected). "
              "No labels will be fabricated or reconstructed.")

    # Backfill window_id for the existing file if it predates that column,
    # so every row in the final dataset is traceable — existing predictions,
    # confidences, and labels are left untouched.
    if 'window_id' not in existing_df.columns:
        existing_df = existing_df.copy()
        existing_df['window_id'] = [
            f"{row.subject}_{i}"
            for i, row in enumerate(existing_df.itertuples(index=False))
        ]
        print(f"  Note: {EXISTING_RESULTS_CSV} had no window_id column — backfilled "
              "one from subject + row order within the file (existing prediction "
              "data was not modified).")

    print(f"  Loaded existing results: {EXISTING_RESULTS_CSV} "
          f"({len(existing_df):,} windows, subjects {sorted(existing_df['subject'].unique())})")
    pieces.append(existing_df)
else:
    print(f"  WARNING: {EXISTING_RESULTS_CSV} not found — existing 4-subject results are missing.")

if os.path.exists(REMAINING_RESULTS_CSV):
    remaining_df = pd.read_csv(REMAINING_RESULTS_CSV)
    print(f"  Loaded new results: {REMAINING_RESULTS_CSV} "
          f"({len(remaining_df):,} windows, subjects {sorted(remaining_df['subject'].unique())})")
    pieces.append(remaining_df)
else:
    print(f"  Note: {REMAINING_RESULTS_CSV} not found (no new run yet, or this run produced no results).")

if not pieces:
    raise SystemExit("No per-window result files found. Nothing to combine.")

# Concatenate all rows directly from both files — every evaluated window is
# used exactly once. No deduplication: two different windows can legitimately
# share identical prediction/confidence values, and window_id already makes
# every row traceable without needing to collapse "duplicates".
final_combined = pd.concat(pieces, ignore_index=True)

# ── SUBJECT COMPLETENESS CHECK ─────────────────────────────────────────────
expected_subjects = set(ALL_NINE_SUBJECTS)
present_subjects_set = set(final_combined['subject'].unique().tolist())
present_subjects = sorted(present_subjects_set)

if present_subjects_set == expected_subjects:
    print(f"\nFINAL VALIDATION: 9 subjects confirmed")
    print(f"{', '.join(str(s) for s in present_subjects)}")
else:
    missing_subjects = sorted(expected_subjects - present_subjects_set)
    extra_subjects = sorted(present_subjects_set - expected_subjects)
    print(f"\nSubject check FAILED.")
    print(f"  Present:  {present_subjects}")
    if missing_subjects:
        print(f"  Missing:  {missing_subjects}")
    if extra_subjects:
        print(f"  Unexpected (not in the 9-subject list): {extra_subjects}")
    raise SystemExit(
        "Stopping before generating final results — set(final_results['subject']) "
        f"!= expected_subjects ({sorted(expected_subjects)}). "
        "Run the remaining subjects, then re-run the combine stage."
    )

final_combined.to_csv(FINAL_RESULTS_CSV, index=False)
print(f"\nSaved combined 9-subject per-window results: {FINAL_RESULTS_CSV} "
      f"({len(final_combined):,} windows)")

combined = final_combined  # everything below recomputes from this combined dataset

# ── PER-SUBJECT SUMMARY (recomputed from combined per-window data) ────────
print(f"\n{'='*65}")
print("PER-SUBJECT SUMMARY (final, all 9 subjects)")
print(f"{'='*65}")

per_subject_rows = []
for sid in present_subjects:
    subj_data = combined[combined['subject'] == sid]
    acc = subj_data['is_correct'].mean()
    row = {"subject": sid, "accuracy": round(acc, 4), "n_windows": len(subj_data)}
    for tl in ['GREEN', 'YELLOW', 'RED']:
        tsub = subj_data[subj_data['traffic_light'] == tl]
        n = len(tsub)
        row[f"{tl.lower()}_accuracy"] = round(tsub['is_correct'].mean(), 4) if n > 0 else None
        row[f"{tl.lower()}_n"] = n
    per_subject_rows.append(row)

per_subject_df = pd.DataFrame(per_subject_rows)
print(per_subject_df.to_string(index=False))
per_subject_df.to_csv("pamap2_per_subject_summary_final.csv", index=False)

# ── RELIABILITY-TIER ANALYSIS ───────────────────────────────────────────────
print(f"\n{'='*65}")
print("COMBINED RELIABILITY-TIER ANALYSIS (final, all 9 subjects)")
print(f"{'='*65}")

tier_rows = []
for tl in ['GREEN', 'YELLOW', 'RED']:
    subset = combined[combined['traffic_light'] == tl]
    n = len(subset)
    pct_total = n / len(combined) if len(combined) > 0 else 0
    tier_acc = subset['is_correct'].mean() if n > 0 else np.nan
    err_rate = 1 - tier_acc if n > 0 else np.nan
    mean_conf = subset['confidence'].mean() if n > 0 else np.nan
    tier_rows.append({
        "tier": tl, "windows": n, "pct_total": round(pct_total, 4),
        "accuracy": round(tier_acc, 4) if n > 0 else None,
        "error_rate": round(err_rate, 4) if n > 0 else None,
        "mean_confidence": round(mean_conf, 4) if n > 0 else None,
    })

tier_df = pd.DataFrame(tier_rows)
tier_df.to_csv("pamap2_reliability_tier_summary_final.csv", index=False)
print(tier_df.to_string(index=False))

# ── ORDERING CHECK: GREEN accuracy > YELLOW accuracy > RED accuracy? ───────
tier_acc_map = {row["tier"]: row["accuracy"] for row in tier_rows}
green_acc, yellow_acc, red_acc = tier_acc_map["GREEN"], tier_acc_map["YELLOW"], tier_acc_map["RED"]
if None in (green_acc, yellow_acc, red_acc):
    ordering_ok = "NO (one or more tiers has no samples)"
else:
    ordering_ok = "YES" if (green_acc > yellow_acc > red_acc) else "NO"
print(f"\nGREEN accuracy > YELLOW accuracy > RED accuracy? {ordering_ok}")
if isinstance(green_acc, float) and isinstance(yellow_acc, float) and isinstance(red_acc, float):
    print(f"  GREEN:  {green_acc:.4f}")
    print(f"  YELLOW: {yellow_acc:.4f}")
    print(f"  RED:    {red_acc:.4f}")

# ── CONFIDENCE-BIN ANALYSIS ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print("CONFIDENCE-BIN ANALYSIS (10 bins, final)")
print(f"{'='*65}")

bin_edges = np.linspace(0, 1, 11)

bin_rows = []       # full precision — used for ECE and plotting
bin_rows_csv = []   # rounded — used only for the saved CSV / printout
for i in range(10):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    mask = (combined['confidence'] >= lo) & (combined['confidence'] < hi if i < 9 else combined['confidence'] <= hi)
    subset = combined[mask]
    n = len(subset)
    mean_conf = subset['confidence'].mean() if n > 0 else np.nan
    acc = subset['is_correct'].mean() if n > 0 else np.nan
    err = 1 - acc if n > 0 else np.nan
    bin_rows.append({
        "bin_low": lo, "bin_high": hi,
        "n_samples": n,
        "mean_confidence": mean_conf if n > 0 else None,
        "accuracy": acc if n > 0 else None,
        "error_rate": err if n > 0 else None,
    })
    bin_rows_csv.append({
        "bin_low": round(lo, 2), "bin_high": round(hi, 2),
        "n_samples": n,
        "mean_confidence": round(mean_conf, 4) if n > 0 else None,
        "accuracy": round(acc, 4) if n > 0 else None,
        "error_rate": round(err, 4) if n > 0 else None,
    })

bin_df = pd.DataFrame(bin_rows_csv)
bin_df.to_csv("pamap2_confidence_bins_final.csv", index=False)
print(bin_df.to_string(index=False))

# ── SPEARMAN CORRELATION ────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SPEARMAN CORRELATION: confidence vs is_correct (final)")
print(f"{'='*65}")
rho, p_value = spearmanr(combined['confidence'], combined['is_correct'])
print(f"  Spearman rho: {rho:.4f}")
print(f"  p-value:      {p_value:.6g}")
print("  NOTE: windows overlap 50% (STEP_SIZE=50 within WINDOW_SIZE=100), so")
print("  consecutive samples are not independent. The p-value assumes i.i.d.")
print("  samples and is therefore NOT a valid basis for a significance claim")
print("  here — reported for reference only. Interpret rho's sign and")
print("  magnitude directly instead.")

# ── EXPECTED CALIBRATION ERROR (ECE) ────────────────────────────────────────
print(f"\n{'='*65}")
print("EXPECTED CALIBRATION ERROR (ECE, final)")
print(f"{'='*65}")

ece = 0.0
n_total = len(combined)
for row in bin_rows:   # full-precision values, not the rounded CSV copies
    n = row["n_samples"]
    if n == 0:
        continue
    ece += (n / n_total) * abs(row["accuracy"] - row["mean_confidence"])
print(f"  ECE: {ece:.4f}")

# ── OVERALL COMBINED METRICS ─────────────────────────────────────────────
print(f"\n{'='*65}")
print("OVERALL COMBINED VALIDATION METRICS (final, all 9 subjects)")
print(f"{'='*65}")

# Accuracy only requires is_correct, which every row has regardless of
# whether activity-label columns are present.
overall_acc = combined['is_correct'].mean()

# Balanced Accuracy is per-class recall averaged across classes, which needs
# true/predicted labels — computed alongside precision/recall/F1 below when
# labels are available for all 9 subjects; otherwise omitted.
overall_bac = None

labels_available = (
    'actual_activity' in combined.columns and
    'predicted_activity' in combined.columns and
    combined['actual_activity'].notna().all() and
    combined['predicted_activity'].notna().all()
)

overall_prec = overall_rec = overall_f1 = None

if labels_available:
    inv_map = {v: k for k, v in ACTIVITY_MAP.items()}
    y_true_all = combined['actual_activity'].map(inv_map).values
    y_pred_all = combined['predicted_activity'].map(inv_map).values

    if pd.isna(y_true_all).any() or pd.isna(y_pred_all).any():
        print("  Note: some activity-label values did not map to a known activityID — "
              "macro Precision/Recall/F1 and label-based Balanced Accuracy omitted.")
        labels_available = False
    else:
        overall_bac = balanced_accuracy_score(y_true_all, y_pred_all)
        overall_prec = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        overall_rec = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        overall_f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)

if not labels_available:
    print("  Note: predicted_activity/actual_activity not available for all 9 subjects — "
        "macro Precision/Recall/F1 omitted. Balanced Accuracy also requires these "
        "labels and is omitted for the same reason; Accuracy, tier accuracy, "
        "confidence bins, ECE, and Spearman rho are unaffected.")

print(f"  Accuracy:            {overall_acc:.4f}")
if overall_bac is not None:
    print(f"  Balanced Accuracy:   {overall_bac:.4f}")
else:
    print(f"  Balanced Accuracy:   omitted (activity labels not available for all 9 subjects)")
if overall_prec is not None:
    print(f"  Precision (macro):   {overall_prec:.4f}")
    print(f"  Recall (macro):      {overall_rec:.4f}")
    print(f"  F1 (macro):          {overall_f1:.4f}")
else:
    print(f"  Precision (macro):   omitted (activity labels not available for all 9 subjects)")
    print(f"  Recall (macro):      omitted (activity labels not available for all 9 subjects)")
    print(f"  F1 (macro):          omitted (activity labels not available for all 9 subjects)")
print(f"  Mean confidence:     {combined['confidence'].mean():.4f}")
print(f"  ECE:                 {ece:.4f}")
print(f"  Spearman rho:        {rho:.4f}  (p={p_value:.4g}, not a valid significance test — overlapping windows)")

final_metrics_row = {
    "accuracy": round(overall_acc, 4),
    "balanced_accuracy": round(overall_bac, 4) if overall_bac is not None else None,
    "precision_macro": round(overall_prec, 4) if overall_prec is not None else None,
    "recall_macro": round(overall_rec, 4) if overall_rec is not None else None,
    "f1_macro": round(overall_f1, 4) if overall_f1 is not None else None,
    "mean_confidence": round(combined['confidence'].mean(), 4),
    "ece": round(ece, 4),
    "spearman_rho": round(rho, 4),
    "spearman_p": p_value,
    "n_windows": n_total,
    "validate_subjects": str(present_subjects),
    "labels_available_all_9_subjects": labels_available,
}
pd.DataFrame([final_metrics_row]).to_csv("pamap2_overall_reliability_metrics_final.csv", index=False)

# ── GRAPH 1: RELIABILITY / CALIBRATION CURVE ────────────────────────────────
plot_bins = [r for r in bin_rows if r["n_samples"] > 0]
bin_centers = [(r["bin_low"] + r["bin_high"]) / 2 for r in plot_bins]
bin_accs = [r["accuracy"] for r in plot_bins]

plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal (accuracy = confidence)')
plt.plot(bin_centers, bin_accs, marker='o', color='steelblue', label='Observed accuracy')
plt.xlabel("Confidence bin (center)")
plt.ylabel("Observed accuracy")
plt.title("Reliability / Calibration Curve — Final VotingClassifier (9 subjects)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graph_18_reliability_calibration_final.png", dpi=150)
plt.close()
print("\nSaved: graph_18_reliability_calibration_final.png")

# ── GRAPH 2: TRAFFIC-LIGHT ACCURACY BAR CHART ───────────────────────────────
tier_labels = tier_df['tier'].tolist()
tier_accs = [a if a is not None else 0 for a in tier_df['accuracy'].tolist()]
colors = {'GREEN': '#2ecc71', 'YELLOW': '#f1c40f', 'RED': '#e74c3c'}

plt.figure(figsize=(6, 6))
bars = plt.bar(tier_labels, tier_accs, color=[colors[t] for t in tier_labels])
for bar, acc in zip(bars, tier_accs):
    plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.1%}",
              ha='center', va='bottom')
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.title("Traffic-Light Tier vs Actual Accuracy (9 subjects)")
plt.tight_layout()
plt.savefig("graph_19_traffic_light_accuracy_final.png", dpi=150)
plt.close()
print("Saved: graph_19_traffic_light_accuracy_final.png")

print(f"\nSaved: {FINAL_RESULTS_CSV}")
print(f"Saved: pamap2_per_subject_summary_final.csv")
print(f"Saved: pamap2_reliability_tier_summary_final.csv")
print(f"Saved: pamap2_confidence_bins_final.csv")
print(f"Saved: pamap2_overall_reliability_metrics_final.csv")
print(f"\n({n_total:,} windows across subjects {present_subjects} — FINAL 9-subject reliability validation)")
print(f"\nExisting 4-subject file ({EXISTING_RESULTS_CSV}) was NOT modified.")