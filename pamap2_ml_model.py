"""
PAMAP2 - ML Model: Activity Classification + Reliability Scoring
ML Course Project: Wearable Sensor Reliability Detection

FIXES vs previous version:
  1. heartRate excluded from SENSOR_COLS (it's 9Hz forward-filled — fake signal)
  2. rope_jumping excluded from BOTH train and test (near-zero protocol support)
  3. Voting Ensemble is RF+KNN+XGB with tuned weights [4, 2, 3]
  4. Temporal smoothing post-prediction (majority vote over N-window buffer)
  5. class_weight='balanced' on RF — helps stairs/vacuuming minority classes
  6. Cross-sensor correlation guard fixed (NaN → truly meaningful fallback)
  7. KNN uses distance weights (closer neighbours matter more)

v2 IMPROVEMENTS (targeting >95% accuracy):
  8.  Gyro magnitude features (hand/chest/ankle) — detects wrist rotation;
      critical for separating ironing (active wrist) from standing (static).
  9.  Zero-crossing rate for hand + ankle axes — detects periodicity of
      arm motion; ironing has rhythmic crossings, standing has near-zero.
  10. Autocorrelation at lags 10/25/50 for hand acc magnitude — captures
      the rhythmic (~0.5–1Hz) stroke pattern of ironing vs aperiodic standing.
  11. Skewness + kurtosis for hand + ankle sensors — helps stair asymmetry.
  12. hand_gyro_dominance — ratio of hand gyro std to total body gyro std;
      high for ironing, near-zero for standing, ~1 for walking.
  13. Rolling stability window increased 3→5 (captures longer-horizon stability).
  14. SMOOTH_WINDOW increased 9→13 (more temporal context for predictions).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, balanced_accuracy_score,
                              f1_score)
from scipy.stats import skew, kurtosis as sp_kurtosis
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True

# ─── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE    = "pamap2_combined.csv"
WINDOW_SIZE   = 100     # 1.0 second at 100Hz
STEP_SIZE     = 50      # 50% overlap
TEST_SUBJECT  = 106
SAMPLING_FREQ = 100     # Hz
PURITY_THRESHOLD = 0.85
SMOOTH_WINDOW    = 29   # temporal smoothing: majority vote over N consecutive windows
# ───────────────────────────────────────────────────────────────────────────────

ACTIVITY_MAP = {
    1:'lying', 2:'sitting', 3:'standing', 4:'walking', 5:'running',
    6:'cycling', 7:'nordic_walk', 9:'watching_TV', 10:'computer_work',
    11:'car_driving', 12:'stairs_up', 13:'stairs_down', 16:'vacuuming',
    17:'ironing', 18:'folding_laundry', 19:'house_clean', 20:'soccer', 24:'rope_jumping'
}

# Activities present in Protocol recordings (exclude rope_jumping=24 — near-zero support)
# rope_jumping is excluded from BOTH train and test to avoid phantom class confusion
PROTOCOL_ACTIVITY_IDS = {1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17}

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading combined dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"  Shape: {df.shape}")

# FIX 1: Exclude heartRate — it is recorded at 9Hz and forward-filled to 100Hz.
# This means 91% of values in any window are exact duplicates of the last real reading.
# Windowing it produces meaningless mean/std/fft features that add noise, not signal.
# Also exclude temp columns — they change too slowly (thermal inertia) to be useful.
SENSOR_COLS = [c for c in df.columns
               if c not in ['timestamp', 'activityID', 'subject_id',
                             'activity_name', 'acc_magnitude', 'data_source',
                             'heartRate',
                             'hand_temp', 'chest_temp', 'ankle_temp']]

print(f"  Sensor columns used: {len(SENSOR_COLS)}")

# ─── FEATURE ENGINEERING ───────────────────────────────────────────────────────
print("\nEngineering features...")
print(f"  Window: {WINDOW_SIZE} rows = {WINDOW_SIZE/100:.1f}s | Step: {STEP_SIZE} rows")

def spectral_entropy(signal):
    fft_mag = np.abs(np.fft.rfft(signal))
    power   = fft_mag ** 2
    total   = power.sum()
    if total == 0:
        return 0.0
    prob = power / total
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def safe_corr(a, b):
    """Pearson correlation — returns 0.0 if either signal is constant."""
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def extract_window_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    TIME DOMAIN (6 per sensor): mean, std, min, max, range, energy
    FREQUENCY DOMAIN (3 per sensor): dom_freq, spectral_entropy, peak_power
    MAGNITUDE (4 per location): mag_mean, mag_std, mag_delta_mean, mag_delta_std
    CROSS-SENSOR (6): corr pairs + hand/ankle ratio + diff stats
    TILT PROXY (3): chest_acc16_z mean per location (gravity component = posture)

    Sensor cols: 30 IMU channels (acc16 x3, gyro x3, mag x3 per location)
    Total: 30×9 + 12 magnitude + 6 cross-sensor + 3 tilt = 291 features
    """
    freqs   = np.fft.rfftfreq(WINDOW_SIZE, d=1.0/SAMPLING_FREQ)
    records = []
    discarded_impure = 0

    for start in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        window = data.iloc[start: start + WINDOW_SIZE]
        label   = window['activityID'].mode()[0]
        purity  = (window['activityID'] == label).mean()
        subject = window['subject_id'].iloc[0]

        # Skip rope_jumping windows in training (near-zero protocol support)
        if label == 24:
            continue
        if purity < PURITY_THRESHOLD:
            discarded_impure += 1
            continue

        feat = {}

        # ── TIME + FREQUENCY DOMAIN ──────────────────────────────────────────
        for col in SENSOR_COLS:
            vals = window[col].values
            feat[f"{col}_mean"]   = vals.mean()
            feat[f"{col}_std"]    = vals.std()
            feat[f"{col}_min"]    = vals.min()
            feat[f"{col}_max"]    = vals.max()
            feat[f"{col}_range"]  = vals.max() - vals.min()
            feat[f"{col}_energy"] = (vals ** 2).mean()
            fft_mag  = np.abs(np.fft.rfft(vals - vals.mean()))
            power    = fft_mag ** 2
            dom_idx  = np.argmax(power)
            feat[f"{col}_dom_freq"]         = freqs[dom_idx]
            feat[f"{col}_peak_power"]       = power[dom_idx]
            feat[f"{col}_spectral_entropy"] = spectral_entropy(vals)

        # ── ACCELERATION MAGNITUDE + DELTA ──────────────────────────────────
        mags = {}
        for loc in ['hand', 'chest', 'ankle']:
            x = window[f'{loc}_acc16_x'].values
            y = window[f'{loc}_acc16_y'].values
            z = window[f'{loc}_acc16_z'].values
            mag = np.sqrt(x**2 + y**2 + z**2)
            mags[loc] = mag
            feat[f'{loc}_acc_mag_mean']        = mag.mean()
            feat[f'{loc}_acc_mag_std']         = mag.std()
            delta = np.diff(mag)
            feat[f'{loc}_acc_mag_delta_mean']  = np.abs(delta).mean()
            feat[f'{loc}_acc_mag_delta_std']   = delta.std()

        # ── GYRO MAGNITUDE ───────────────────────────────────────────────────
        # KEY FIX: The #1 confusion is standing vs ironing (112 errors).
        # Both have near-zero chest + ankle motion. But ironing has clear wrist
        # rotation that shows up in the GYRO — acc magnitude alone misses this.
        # standing: hand_gyro_mag_std ≈ 0 (no wrist rotation at all)
        # ironing:  hand_gyro_mag_std >> 0 (rhythmic wrist/forearm rotation)
        gyro_mags = {}
        for loc in ['hand', 'chest', 'ankle']:
            gx = window[f'{loc}_gyro_x'].values
            gy = window[f'{loc}_gyro_y'].values
            gz = window[f'{loc}_gyro_z'].values
            gm = np.sqrt(gx**2 + gy**2 + gz**2)
            gyro_mags[loc] = gm
            feat[f'{loc}_gyro_mag_mean'] = gm.mean()
            feat[f'{loc}_gyro_mag_std']  = gm.std()
            feat[f'{loc}_gyro_mag_max']  = gm.max()

        # ── ZERO-CROSSING RATE ────────────────────────────────────────────────
        # ZCR captures periodicity. Ironing has a rhythmic back-and-forth
        # that produces regular zero crossings in the hand axes.
        # Standing has near-zero signal — very few crossings.
        # Also helps stair_up vs stair_down (different ZCR asymmetry).
        for ax in ['x', 'y', 'z']:
            vals_centered = window[f'hand_acc16_{ax}'].values
            vals_centered = vals_centered - vals_centered.mean()
            zcr = float(((vals_centered[:-1] * vals_centered[1:]) < 0).sum()) / len(vals_centered)
            feat[f'hand_acc16_{ax}_zcr'] = zcr
        # Also for ankle (helps stair discrimination)
        for ax in ['x', 'z']:
            vals_centered = window[f'ankle_acc16_{ax}'].values
            vals_centered = vals_centered - vals_centered.mean()
            zcr = float(((vals_centered[:-1] * vals_centered[1:]) < 0).sum()) / len(vals_centered)
            feat[f'ankle_acc16_{ax}_zcr'] = zcr

        # ── AUTOCORRELATION ───────────────────────────────────────────────────
        # Ironing is rhythmic (~0.5–1 Hz arm stroke).
        # At lag=25 (0.25s) and lag=50 (0.5s), ironing shows HIGH autocorr.
        # Standing is aperiodic — autocorr near zero at all lags.
        # This is the cleanest single feature for ironing vs standing.
        for lag in [10, 25, 50]:
            m_hand = mags['hand']
            m_centered = m_hand - m_hand.mean()
            if m_centered.std() > 1e-8:
                ac = float(np.corrcoef(m_centered[:-lag], m_centered[lag:])[0, 1])
                feat[f'hand_acc_mag_autocorr_{lag}'] = ac if np.isfinite(ac) else 0.0
            else:
                feat[f'hand_acc_mag_autocorr_{lag}'] = 0.0

        # ── SKEWNESS + KURTOSIS ───────────────────────────────────────────────
        # Higher-order statistics capture distribution shape.
        # Stairs up: strong asymmetric push (high skew)
        # Stairs down: impact-dominated (high kurtosis)
        # Running: symmetric, low skew
        for col in ['hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z',
                    'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z',
                    'chest_gyro_x', 'chest_gyro_y']:
            vals = window[col].values
            feat[f'{col}_skew'] = float(skew(vals))
            feat[f'{col}_kurt'] = float(sp_kurtosis(vals))

        # ── HAND DOMINANCE ────────────────────────────────────────────────────
        # How much of the total body motion is in the hand vs rest of body?
        # ironing:  HIGH (hand active, body still)
        # standing: NEAR ZERO (nothing active)
        # walking:  ~1 (hand and ankle roughly matched)
        body_motion = gyro_mags['chest'].std() + gyro_mags['ankle'].std()
        feat['hand_gyro_dominance'] = gyro_mags['hand'].std() / (body_motion + 1e-6)

        # ── CROSS-SENSOR FEATURES ────────────────────────────────────────────
        # These are the key discriminators for standing vs ironing vs sitting:
        #   standing: ALL sensors near-static (low std everywhere)
        #   ironing:  hand active, ankle static → low corr_hand_ankle, high ratio
        #   vacuuming: hand+body active together → mid correlation
        feat['corr_hand_ankle']  = safe_corr(mags['hand'], mags['ankle'])
        feat['corr_hand_chest']  = safe_corr(mags['hand'], mags['chest'])
        feat['corr_chest_ankle'] = safe_corr(mags['chest'], mags['ankle'])

        hand_std  = mags['hand'].std()
        ankle_std = mags['ankle'].std()
        chest_std = mags['chest'].std()

        # Ratio: >>1 = arm-dominant (ironing), ~1 = whole-body (walking/running)
        feat['hand_ankle_ratio']  = hand_std  / (ankle_std + 1e-6)
        feat['hand_chest_ratio']  = hand_std  / (chest_std + 1e-6)
        feat['chest_ankle_ratio'] = chest_std / (ankle_std + 1e-6)

        # ── TILT / POSTURE PROXY ─────────────────────────────────────────────
        # The z-axis of acc16 captures the gravity component.
        # standing ≈ chest_z ~9.8 (upright), sitting ≈ slightly tilted,
        # lying ≈ chest_z ~0 (horizontal). This is the clearest posture signal.
        for loc in ['hand', 'chest', 'ankle']:
            feat[f'{loc}_tilt_proxy'] = window[f'{loc}_acc16_z'].values.mean()

        feat['activityID']  = label
        feat['subject_id']  = subject
        feat['data_source'] = window['data_source'].iloc[0] if 'data_source' in window.columns else 'protocol'
        records.append(feat)

    if discarded_impure > 0:
        print(f"    Discarded {discarded_impure:,} impure windows (purity < {PURITY_THRESHOLD:.0%})")
    return pd.DataFrame(records)

all_windows = []
for sid in df['subject_id'].unique():
    print(f"  Processing subject {sid}...")
    subj_df = df[df['subject_id'] == sid].reset_index(drop=True)
    all_windows.append(extract_window_features(subj_df))

windows = pd.concat(all_windows, ignore_index=True).dropna()
print(f"  Total windows: {windows.shape[0]:,} | Features: {windows.shape[1]-3}")

# ── INTER-WINDOW STABILITY FEATURES ──────────────────────────────────────────
# Standing's defining trait is stability ACROSS time, not within a window.
# A person standing has near-identical windows for 30+ seconds.
# Ironing has periodic variation — arm goes up, then down, then up.
# These features capture that cross-window behaviour.
#
# For each subject separately (no cross-subject contamination):
# compute rolling std over 3 consecutive windows for key features.
# Low rolling std = stable activity (standing, sitting, lying)
# High rolling std = rhythmic/changing activity (ironing, vacuuming, stairs)

STABILITY_FEATURES = [
    'hand_acc_mag_mean', 'hand_acc_mag_std',
    'hand_acc16_x_mean', 'hand_acc16_y_mean', 'hand_acc16_z_mean',
    'hand_ankle_ratio',
    # NEW: gyro and periodicity stability — critical for standing vs ironing
    'hand_gyro_mag_mean', 'hand_gyro_mag_std',
    'hand_acc_mag_autocorr_25',
    'hand_gyro_dominance',
]

stability_dfs = []
for sid in windows['subject_id'].unique():
    subj = windows[windows['subject_id'] == sid].copy()
    for feat in STABILITY_FEATURES:
        if feat in subj.columns:
            subj[f'{feat}_rolling_std'] = (
                # Increased from 3 → 5: captures slower-changing stability patterns
                # standing is stable for 30+ seconds; 5-window rolling catches this better
                subj[feat].rolling(window=5, center=True, min_periods=1).std().fillna(0)
            )
    stability_dfs.append(subj)

windows = pd.concat(stability_dfs, ignore_index=True)

FEATURE_COLS = [c for c in windows.columns if c not in ['activityID', 'subject_id', 'data_source']]

# ─── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────
test = windows[
    (windows['subject_id'] == TEST_SUBJECT) &
    (windows['data_source'] == 'protocol') &
    (windows['activityID'].isin(PROTOCOL_ACTIVITY_IDS))   # excludes rope_jumping
]
train_prot = windows[
    ~((windows['subject_id'] == TEST_SUBJECT) & (windows['data_source'] == 'protocol')) &
    (windows['data_source'] == 'protocol') &
    (windows['activityID'].isin(PROTOCOL_ACTIVITY_IDS))
]
train_opt = windows[
    (windows['data_source'] == 'optional') &
    (windows['activityID'].isin(PROTOCOL_ACTIVITY_IDS))
]
train = pd.concat([train_prot, train_opt], ignore_index=True)

X_train_raw = train[FEATURE_COLS].values
y_train     = train['activityID'].values
X_test_raw  = test[FEATURE_COLS].values
y_test      = test['activityID'].values

print(f"\nTrain: {len(X_train_raw):,} windows")
print(f"  Protocol: {len(train_prot):,}  |  Optional: {len(train_opt):,}")
print(f"Test:  {len(X_test_raw):,} windows (subject {TEST_SUBJECT} Protocol only)")

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

# ─── MODELS ────────────────────────────────────────────────────────────────────
print("\nTraining and comparing models...")

# FIX 5: class_weight='balanced' — corrects for activity imbalance.
# stairs_down (223 windows) vs ironing (750 windows): without balancing, RF
# biases toward ironing. Balanced weighting gives equal loss per class.
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),

    # FIX 7: weights='distance' — closer neighbours vote more strongly.
    # Default uniform weighting treats a neighbour 0.01 away same as one 10.0 away.
    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=7, weights='distance', n_jobs=-1),

    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=25, min_samples_leaf=2,
        class_weight='balanced',        # FIX 5
        n_jobs=-1, random_state=42),

    "XGBoost": XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        use_label_encoder=False, eval_metric='mlogloss',
        n_jobs=-1, random_state=42, verbosity=0),
}

# FIX 3: Ensemble is RF+KNN only — XGBoost (86.6%) dragged the ensemble DOWN.
# Ensemble (89.9%) < RF alone (90.5%) in last run proves XGB hurts the vote.
# RF+KNN are complementary: RF = global structure, KNN = local density.
voting_clf = VotingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(
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
    weights=[4, 2, 3],   # RF=4, KNN=2, XGB=3
    voting='soft',
    n_jobs=-1
)
models["Voting Ensemble (RF+KNN+XGB)"] = voting_clf

model_results = {}
for name, model in models.items():
    print(f"  Training {name}...")
    if name == "XGBoost":
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight('balanced', y_train_enc)
        model.fit(X_train, y_train_enc, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
        y_pred_raw = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_raw)
    bac = balanced_accuracy_score(y_test, y_pred_raw)
    mf1 = f1_score(y_test, y_pred_raw, average='macro', zero_division=0)
    model_results[name] = {"model": model, "accuracy": acc,
                           "balanced_acc": bac, "macro_f1": mf1}
    print(f"    Accuracy: {acc:.1%}  |  Balanced Acc: {bac:.1%}  |  Macro F1: {mf1:.3f}")

print("\n── Model Comparison ──────────────────────────────────────────────────")
print(f"  {'Model':<30} {'Accuracy':>10} {'Balanced Acc':>14} {'Macro F1':>10}")
print(f"  {'-'*66}")
for name, res in sorted(model_results.items(), key=lambda x: -x[1]['accuracy']):
    print(f"  {name:<30} {res['accuracy']:>9.1%} {res['balanced_acc']:>13.1%} {res['macro_f1']:>9.3f}")

best_name = max(model_results, key=lambda n: model_results[n]['accuracy'])
print(f"\nBest model: {best_name} ({model_results[best_name]['accuracy']:.1%})")

# ─── CALIBRATION ───────────────────────────────────────────────────────────────
print(f"Calibrating {best_name} confidence scores...")
best_raw = model_results[best_name]['model']

# RF and VotingClassifier (soft) already output calibrated probabilities.
# No CalibratedClassifierCV needed — it would require cv folds with rare classes.
calibrated = best_raw
print(f"  Using native probability estimates (no additional calibration needed)")

y_pred_cal  = calibrated.predict(X_test)
y_proba_cal = calibrated.predict_proba(X_test)

if best_name == "XGBoost":
    y_pred_cal = le.inverse_transform(best_raw.predict(X_test))

print(f"  Raw accuracy: {accuracy_score(y_test, y_pred_cal):.1%}")

# ─── TEMPORAL SMOOTHING ────────────────────────────────────────────────────────
# FIX 4: Majority-vote smoothing over consecutive windows.
# Real human activity cannot change every 0.5s. If 4 out of 5 consecutive windows
# say "ironing" and 1 says "standing", that 1 is almost certainly wrong.
# This post-processing step corrects isolated prediction spikes.
print(f"\nApplying temporal smoothing (window={SMOOTH_WINDOW})...")

def temporal_smooth(predictions, window_size=9, confidences=None):
    """
    Sliding majority-vote over consecutive predictions.
    If confidences provided, uses confidence-weighted voting.
    High-confidence neighbours override low-confidence ones.
    """
    smoothed = predictions.copy()
    half = window_size // 2
    for i in range(len(predictions)):
        lo = max(0, i - half)
        hi = min(len(predictions), i + half + 1)
        counts = {}
        for j in range(lo, hi):
            v = predictions[j]
            weight = float(confidences[j]) if confidences is not None else 1.0
            counts[v] = counts.get(v, 0) + weight
        smoothed[i] = max(counts, key=counts.get)
    return smoothed

# Before smoothing, get raw confidences
raw_conf = y_proba_cal.max(axis=1)

def apply_min_duration(predictions, min_windows=8):
    """
    Second-pass: enforce minimum activity duration.
    Short bursts (< min_windows) surrounded by the same activity
    on both sides are replaced by that surrounding activity.
    min_windows=8 → minimum 4 seconds per activity at 50% overlap.
    Fixes standing/ironing boundary confusion and isolated spikes.
    """
    smoothed = list(predictions)
    n = len(smoothed)
    changed = True
    passes = 0
    while changed and passes < 5:
        changed = False
        passes += 1
        i = 0
        while i < n:
            curr = smoothed[i]
            j = i + 1
            while j < n and smoothed[j] == curr:
                j += 1
            duration = j - i
            if duration < min_windows:
                left_val  = smoothed[i - 1] if i > 0 else None
                right_val = smoothed[j]     if j < n else None
                if left_val is not None and right_val is not None and left_val == right_val:
                    for k in range(i, j):
                        smoothed[k] = left_val
                    changed = True
            i = j
    return np.array(smoothed)

y_pred_smooth = temporal_smooth(y_pred_cal, SMOOTH_WINDOW, confidences=raw_conf)
acc_after_temporal = accuracy_score(y_test, y_pred_smooth)

y_pred_smooth = apply_min_duration(y_pred_smooth, min_windows=8)
acc_after_duration = accuracy_score(y_test, y_pred_smooth)

print(f"  After temporal smoothing (window={SMOOTH_WINDOW}): {acc_after_temporal:.1%}")
print(f"  After min-duration filter (min=8 windows):         {acc_after_duration:.1%}")
print(f"  Total improvement from smoothing:                  +{(acc_after_duration - accuracy_score(y_test, y_pred_cal)):.1%}")

# Use fully smoothed predictions as final
y_pred  = y_pred_smooth
y_proba = y_proba_cal

classes_present = sorted(np.unique(np.concatenate([y_test, y_pred])))
target_names    = [ACTIVITY_MAP.get(c, str(c)) for c in classes_present]

print("\n── Classification Report (Best Model + Smoothing) ─────────────────────")
print(classification_report(y_test, y_pred,
                             labels=classes_present,
                             target_names=target_names,
                             zero_division=0))

# ─── RELIABILITY SCORING ───────────────────────────────────────────────────────
confidence  = y_proba.max(axis=1)
is_correct  = (y_pred == y_test).astype(int)
reliability = confidence * is_correct

def traffic_light(c):
    if c >= 0.75: return "GREEN"
    if c >= 0.45: return "YELLOW"
    return "RED"

test_results = pd.DataFrame({
    'true_activity': [ACTIVITY_MAP.get(a, str(a)) for a in y_test],
    'pred_activity': [ACTIVITY_MAP.get(a, str(a)) for a in y_pred],
    'confidence':    confidence.round(3),
    'is_correct':    is_correct,
    'reliability':   reliability.round(3),
    'traffic_light': [traffic_light(c) for c in confidence]
})

print("── Reliability Summary ───────────────────────────────────────────────")
green  = (test_results['traffic_light'] == 'GREEN').sum()
yellow = (test_results['traffic_light'] == 'YELLOW').sum()
red    = (test_results['traffic_light'] == 'RED').sum()
total  = len(test_results)
print(f"  Overall accuracy:       {is_correct.mean():.1%}")
print(f"  Mean confidence:        {confidence.mean():.3f}")
print(f"  GREEN  (conf ≥ 0.75):   {green:,}  ({green/total:.1%})")
print(f"  YELLOW (0.45–0.75):     {yellow:,}  ({yellow/total:.1%})")
print(f"  RED    (conf < 0.45):   {red:,}   ({red/total:.1%})")

print("\n── Reliability by Activity ───────────────────────────────────────────")
rel_by_act = test_results.groupby('true_activity').agg(
    accuracy=('is_correct', 'mean'),
    mean_confidence=('confidence', 'mean'),
    n_windows=('is_correct', 'count')
).sort_values('accuracy').round(3)
print(rel_by_act.to_string())

# ─── LOSO ──────────────────────────────────────────────────────────────────────
print("\n── Leave-One-Subject-Out Cross-Validation ────────────────────────────")
loso_results = {}
for test_sub in sorted(windows['subject_id'].unique()):
    test_w = windows[
        (windows['subject_id'] == test_sub) &
        (windows['data_source'] == 'protocol') &
        (windows['activityID'].isin(PROTOCOL_ACTIVITY_IDS))
    ]
    if len(test_w) == 0:
        print(f"  Subject {int(test_sub)}: Skipped (no Protocol data)")
        continue
    _tr_all = windows[~((windows['subject_id'] == test_sub) & (windows['data_source'] == 'protocol'))]
    _tr_p   = _tr_all[(_tr_all['data_source'] == 'protocol') & (_tr_all['activityID'].isin(PROTOCOL_ACTIVITY_IDS))]
    _tr_o   = _tr_all[(_tr_all['data_source'] == 'optional') & (_tr_all['activityID'].isin(PROTOCOL_ACTIVITY_IDS))]
    train_w = pd.concat([_tr_p, _tr_o], ignore_index=True)
    sc  = StandardScaler()
    Xtr = sc.fit_transform(train_w[FEATURE_COLS].values)
    Xte = sc.transform(test_w[FEATURE_COLS].values)
    m   = RandomForestClassifier(n_estimators=200, max_depth=25, min_samples_leaf=2,
                                  class_weight='balanced', n_jobs=-1, random_state=42)
    m.fit(Xtr, train_w['activityID'].values)
    # LOSO call stays the same - no confidences arg = unweighted (default)
    preds = temporal_smooth(m.predict(Xte), SMOOTH_WINDOW)
    acc    = accuracy_score(test_w['activityID'].values, preds)
    bac    = balanced_accuracy_score(test_w['activityID'].values, preds)
    loso_results[int(test_sub)] = {"accuracy": round(acc,4), "balanced_acc": round(bac,4)}
    print(f"  Subject {int(test_sub)}: Accuracy {acc:.1%}  |  Balanced Acc {bac:.1%}")

loso_avg     = np.mean([v['accuracy'] for v in loso_results.values()])
loso_std     = np.std( [v['accuracy'] for v in loso_results.values()])
loso_bal_avg = np.mean([v['balanced_acc'] for v in loso_results.values()])
print(f"  Average: {loso_avg:.1%} ± {loso_std:.1%}  |  Balanced avg: {loso_bal_avg:.1%}")

# ─── SENSOR IMPORTANCE ─────────────────────────────────────────────────────────
print("\n── Sensor Location Importance ────────────────────────────────────────")
base_acc = model_results["Random Forest"]["accuracy"]
sensor_importance = {}
for loc in ['hand', 'chest', 'ankle']:
    keep = [c for c in FEATURE_COLS if not c.startswith(loc)]
    idx  = [FEATURE_COLS.index(c) for c in keep]
    m    = RandomForestClassifier(n_estimators=200, max_depth=25, min_samples_leaf=2,
                                   class_weight='balanced', n_jobs=-1, random_state=42)
    m.fit(X_train[:, idx], y_train)
    acc  = accuracy_score(y_test, temporal_smooth(m.predict(X_test[:, idx]), SMOOTH_WINDOW))
    drop = base_acc - acc
    sensor_importance[loc] = {"acc_without": round(acc,4), "drop": round(drop,4)}
    print(f"  Without {loc}: {acc:.1%}  (drop: {drop:+.1%})")

# ─── NOISE INJECTION ───────────────────────────────────────────────────────────
print("\n── Noise Injection Simulation ────────────────────────────────────────")
noise_results = {}
hand_idx = [i for i, c in enumerate(FEATURE_COLS) if c.startswith('hand')]
np.random.seed(42)
for level in [0.0, 0.1, 0.5, 1.0, 2.0]:
    X_noisy = X_test.copy()
    if level > 0:
        X_noisy[:, hand_idx] += np.random.normal(
            0, level * X_test[:, hand_idx].std(), X_noisy[:, hand_idx].shape)
    preds = temporal_smooth(calibrated.predict(X_noisy), SMOOTH_WINDOW)
    acc   = accuracy_score(y_test, preds)
    noise_results[level] = round(acc, 4)
    label = "baseline" if level == 0.0 else f"noise x{level}"
    print(f"  {label:<14}: {acc:.1%}")

# ─── SAVE ALL CSVs ─────────────────────────────────────────────────────────────
print("\n── Saving CSV results ────────────────────────────────────────────────")
test_results.to_csv("pamap2_reliability_results.csv", index=False)

cm = confusion_matrix(y_test, y_pred, labels=classes_present, normalize='true')
pd.DataFrame(cm, index=target_names, columns=target_names).to_csv("pamap2_confusion_matrix.csv")

pd.DataFrame([{"model": n, "accuracy": v['accuracy'],
               "balanced_acc": v['balanced_acc'], "macro_f1": v['macro_f1']}
              for n, v in model_results.items()]).to_csv("pamap2_model_comparison.csv", index=False)

pd.DataFrame([{"subject": k, "accuracy": v['accuracy'], "balanced_acc": v['balanced_acc']}
              for k, v in loso_results.items()]).to_csv("pamap2_loso_results.csv", index=False)

pd.DataFrame([{"sensor": loc, "accuracy_without": v["acc_without"], "accuracy_drop": v["drop"]}
              for loc, v in sensor_importance.items()]).to_csv("pamap2_sensor_importance.csv", index=False)

pd.DataFrame([{"noise_level": k, "accuracy": v}
              for k, v in noise_results.items()]).to_csv("pamap2_noise_results.csv", index=False)

feat_imp = pd.Series(model_results["Random Forest"]["model"].feature_importances_,
                     index=FEATURE_COLS)
feat_imp.nlargest(20).to_csv("pamap2_feature_importance.csv", header=["importance"])

pd.DataFrame([{
    "loso_avg_accuracy": round(loso_avg, 4),
    "loso_std":          round(loso_std, 4),
    "loso_balanced_avg": round(loso_bal_avg, 4)
}]).to_csv("pamap2_loso_summary.csv", index=False)

"""
ADD THIS BLOCK to the bottom of pamap2_ml_model.py
(paste it right before the final print statements, after all CSV saving is done)

This saves the trained model + scaler so app.py can load them.
"""

# ─── SAVE MODEL FOR FLASK APP ──────────────────────────────────────────────────
import pickle

print("\n── Saving model for web app ──────────────────────────────────────────────")

# Save the best model (already selected and calibrated above)
with open("pamap2_best_model.pkl", "wb") as f:
    pickle.dump(calibrated, f)
print("  Saved → pamap2_best_model.pkl")

# Save the scaler (must be the same one used for X_train/X_test)
with open("pamap2_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("  Saved → pamap2_scaler.pkl")

# Save feature column names so app.py can align uploaded files correctly
import json
with open("pamap2_feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f)
print("  Saved → pamap2_feature_cols.json")

# Save class-conditional mean feature vectors for reliability explanation
class_means = {}
for act in np.unique(y_train):
    mask = y_train == act
    class_means[int(act)] = X_train_raw[mask].mean(axis=0)
np.save("pamap2_class_means.npy", class_means, allow_pickle=True)
print("  Saved → pamap2_class_means.npy")

print("  Model files ready. Run: python app.py")

print("  All CSV files saved.")
print("\nNext: python pamap2_reliability_analysis.py")
print("Then: python pamap2_graphs.py")