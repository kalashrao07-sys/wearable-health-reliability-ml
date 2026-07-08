"""
PAMAP2 Flask Backend — Sensor Reliability Checker
Run: python app.py  →  open http://localhost:5000

Sensor reliability logic:
  User says "I am walking" + provides sensor readings via sliders.
  Backend synthesises a 100-row window, extracts 264 features,
  runs the real model, gets P(walking). That IS the reliability score.
  High P = sensor matches expected pattern = sensor OK.
  Low  P = readings don't match = sensor faulty/misplaced.
"""

import os, traceback
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from scipy.stats import skew, kurtosis as sp_kurtosis
import pickle, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)   # D:\ML  — where model pkl files live

MODEL_PATH     = os.path.join(PARENT, "pamap2_best_model.pkl")
SCALER_PATH    = os.path.join(PARENT, "pamap2_scaler.pkl")
RESULTS_PATH   = os.path.join(PARENT, "pamap2_reliability_results.csv")
MODEL_CMP_PATH = os.path.join(PARENT, "pamap2_model_comparison.csv")
LOSO_PATH      = os.path.join(PARENT, "pamap2_loso_results.csv")

model, scaler = None, None

def load_artefacts():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
        print(f"  Model  : {type(model).__name__}")
        print(f"  Classes: {list(model.classes_)}")
        print(f"  Scaler features: {len(scaler.feature_names_in_) if hasattr(scaler,'feature_names_in_') else 'unknown'}")
        return True
    print(f"Model not found at: {MODEL_PATH}")
    print("Run pamap2_ml_model.py and add the save_model_snippet first.")
    return False

# ── Constants ─────────────────────────────────────────────────────────────────
ACTIVITY_MAP = {
    1:'lying', 2:'sitting', 3:'standing', 4:'walking', 5:'running',
    6:'cycling', 7:'nordic_walk', 12:'stairs_up', 13:'stairs_down',
    16:'vacuuming', 17:'ironing'
}
ACTIVITY_NAME_TO_ID = {v: k for k, v in ACTIVITY_MAP.items()}
SAMPLING_FREQ = 100
WINDOW_SIZE   = 100

SENSOR_COLS = [
    'hand_acc16_x','hand_acc16_y','hand_acc16_z',
    'hand_gyro_x','hand_gyro_y','hand_gyro_z',
    'hand_mag_x','hand_mag_y','hand_mag_z',
    'chest_acc16_x','chest_acc16_y','chest_acc16_z',
    'chest_gyro_x','chest_gyro_y','chest_gyro_z',
    'chest_mag_x','chest_mag_y','chest_mag_z',
    'ankle_acc16_x','ankle_acc16_y','ankle_acc16_z',
    'ankle_gyro_x','ankle_gyro_y','ankle_gyro_z',
    'ankle_mag_x','ankle_mag_y','ankle_mag_z',
]

# ── Feature extraction — identical to pamap2_ml_model.py ─────────────────────
def spectral_entropy(signal):
    fft_mag = np.abs(np.fft.rfft(signal))
    power   = fft_mag ** 2
    total   = power.sum()
    if total == 0: return 0.0
    prob = power / total
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def safe_corr(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def extract_features(window_df):
    """
    Produces all 308 features matching the trained model.
    Breakdown:
      27 SENSOR_COLS × 9 time/freq stats = 243
      acc_mag (3 locs × 4)               =  12
      gyro_mag (3 locs × 3)  [NEW]       =   9
      ZCR hand x,y,z          [NEW]       =   3
      ZCR ankle x,z           [NEW]       =   2
      autocorr lags 10,25,50  [NEW]       =   3
      skew+kurt 8 channels    [NEW]       =  16
      cross-sensor                        =   6
      tilt proxy                          =   3
      hand_gyro_dominance     [NEW]       =   1
      rolling_std (10 feats)  [NEW→0.0]   =  10
      TOTAL                               = 308
    """
    freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0/SAMPLING_FREQ)
    feat  = {}

    # ── Time + frequency domain ──────────────────────────────────────────────
    for col in SENSOR_COLS:
        if col not in window_df.columns:
            continue
        vals = window_df[col].values.astype(float)
        feat[f"{col}_mean"]             = float(vals.mean())
        feat[f"{col}_std"]              = float(vals.std())
        feat[f"{col}_min"]              = float(vals.min())
        feat[f"{col}_max"]              = float(vals.max())
        feat[f"{col}_range"]            = float(vals.max() - vals.min())
        feat[f"{col}_energy"]           = float((vals ** 2).mean())
        fft_mag = np.abs(np.fft.rfft(vals - vals.mean()))
        power   = fft_mag ** 2
        dom_idx = int(np.argmax(power))
        feat[f"{col}_dom_freq"]         = float(freqs[dom_idx])
        feat[f"{col}_peak_power"]       = float(power[dom_idx])
        feat[f"{col}_spectral_entropy"] = float(spectral_entropy(vals))

    # ── Accelerometer magnitude ──────────────────────────────────────────────
    mags = {}
    for loc in ['hand', 'chest', 'ankle']:
        x   = window_df[f'{loc}_acc16_x'].values.astype(float)
        y   = window_df[f'{loc}_acc16_y'].values.astype(float)
        z   = window_df[f'{loc}_acc16_z'].values.astype(float)
        mag = np.sqrt(x**2 + y**2 + z**2)
        mags[loc] = mag
        feat[f'{loc}_acc_mag_mean']       = float(mag.mean())
        feat[f'{loc}_acc_mag_std']        = float(mag.std())
        delta = np.diff(mag)
        feat[f'{loc}_acc_mag_delta_mean'] = float(np.abs(delta).mean())
        feat[f'{loc}_acc_mag_delta_std']  = float(delta.std())

    # ── Gyro magnitude [NEW] ─────────────────────────────────────────────────
    # Key for separating standing (no wrist rotation) from ironing (active wrist)
    gyro_mags = {}
    for loc in ['hand', 'chest', 'ankle']:
        gx = window_df[f'{loc}_gyro_x'].values.astype(float)
        gy = window_df[f'{loc}_gyro_y'].values.astype(float)
        gz = window_df[f'{loc}_gyro_z'].values.astype(float)
        gm = np.sqrt(gx**2 + gy**2 + gz**2)
        gyro_mags[loc] = gm
        feat[f'{loc}_gyro_mag_mean'] = float(gm.mean())
        feat[f'{loc}_gyro_mag_std']  = float(gm.std())
        feat[f'{loc}_gyro_mag_max']  = float(gm.max())

    # ── Zero-crossing rate [NEW] ─────────────────────────────────────────────
    # Detects periodicity: ironing/walking has regular crossings, standing barely any
    for ax in ['x', 'y', 'z']:
        v = window_df[f'hand_acc16_{ax}'].values.astype(float)
        v = v - v.mean()
        feat[f'hand_acc16_{ax}_zcr'] = float(((v[:-1] * v[1:]) < 0).sum()) / len(v)
    for ax in ['x', 'z']:
        v = window_df[f'ankle_acc16_{ax}'].values.astype(float)
        v = v - v.mean()
        feat[f'ankle_acc16_{ax}_zcr'] = float(((v[:-1] * v[1:]) < 0).sum()) / len(v)

    # ── Autocorrelation [NEW] ────────────────────────────────────────────────
    # Rhythmic activities (ironing, walking) have high autocorr at short lags
    for lag in [10, 25, 50]:
        m  = mags['hand']
        mc = m - m.mean()
        if mc.std() > 1e-8:
            ac = float(np.corrcoef(mc[:-lag], mc[lag:])[0, 1])
            feat[f'hand_acc_mag_autocorr_{lag}'] = ac if np.isfinite(ac) else 0.0
        else:
            feat[f'hand_acc_mag_autocorr_{lag}'] = 0.0

    # ── Skewness + kurtosis [NEW] ────────────────────────────────────────────
    # Captures distribution shape: stairs_up (high skew), stairs_down (high kurt)
    for col in ['hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z',
                'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z',
                'chest_gyro_x', 'chest_gyro_y']:
        if col not in window_df.columns:
            feat[f'{col}_skew'] = 0.0
            feat[f'{col}_kurt'] = 0.0
            continue
        vals = window_df[col].values.astype(float)
        feat[f'{col}_skew'] = float(skew(vals))
        feat[f'{col}_kurt'] = float(sp_kurtosis(vals))

    # ── Cross-sensor features ────────────────────────────────────────────────
    feat['corr_hand_ankle']   = safe_corr(mags['hand'], mags['ankle'])
    feat['corr_hand_chest']   = safe_corr(mags['hand'], mags['chest'])
    feat['corr_chest_ankle']  = safe_corr(mags['chest'], mags['ankle'])
    feat['hand_ankle_ratio']  = float(mags['hand'].std()  / (mags['ankle'].std() + 1e-6))
    feat['hand_chest_ratio']  = float(mags['hand'].std()  / (mags['chest'].std() + 1e-6))
    feat['chest_ankle_ratio'] = float(mags['chest'].std() / (mags['ankle'].std() + 1e-6))

    # ── Hand gyro dominance [NEW] ────────────────────────────────────────────
    # High for ironing (hand dominant), low for standing (all near-zero)
    body_motion = gyro_mags['chest'].std() + gyro_mags['ankle'].std()
    feat['hand_gyro_dominance'] = float(gyro_mags['hand'].std() / (body_motion + 1e-6))

    # ── Tilt proxy ───────────────────────────────────────────────────────────
    for loc in ['hand', 'chest', 'ankle']:
        feat[f'{loc}_tilt_proxy'] = float(window_df[f'{loc}_acc16_z'].values.mean())

    # ── Rolling stability features [NEW → set to 0.0] ────────────────────────
    # These require multiple consecutive windows to compute (multi-window rolling std).
    # For single-window synthesis they are meaningless, so we set them to 0.
    # The model was trained with them; omitting them would cause a column-count mismatch.
    for sf in ['hand_acc_mag_mean', 'hand_acc_mag_std',
               'hand_acc16_x_mean', 'hand_acc16_y_mean', 'hand_acc16_z_mean',
               'hand_ankle_ratio', 'hand_gyro_mag_mean', 'hand_gyro_mag_std',
               'hand_acc_mag_autocorr_25', 'hand_gyro_dominance']:
        feat[f'{sf}_rolling_std'] = 0.0

    return feat

def predict_window(window_df):
    """Extract features, align to scaler, run model. Returns (proba_array, classes_array)."""
    feat    = extract_features(window_df)
    feat_df = pd.DataFrame([feat])

    # Get the exact feature order the scaler/model expects
    if hasattr(scaler, 'feature_names_in_'):
        feat_names = list(scaler.feature_names_in_)
    else:
        feat_names = feat_df.columns.tolist()

    # Fill any missing features with 0 (shouldn't happen, but safe)
    for c in feat_names:
        if c not in feat_df.columns:
            feat_df[c] = 0.0
    feat_df = feat_df[feat_names].fillna(0.0)

    X     = scaler.transform(feat_df.values)
    proba = model.predict_proba(X)[0]
    return proba, model.classes_

def traffic_light(c):
    if c >= 0.75: return "GREEN"
    if c >= 0.45: return "YELLOW"
    return "RED"

# ── Window synthesis ──────────────────────────────────────────────────────────
# Real signal profiles derived from PAMAP2 dataset characteristics
# freq = dominant motion frequency (Hz), amp = acceleration amplitude scale per location
ACTIVITY_PROFILES = {
    'lying':       {'freq': 0.2,  'amp_hand': 0.05, 'amp_chest': 0.02, 'amp_ankle': 0.02, 'gyro_scale': 0.02},
    'sitting':     {'freq': 0.2,  'amp_hand': 0.10, 'amp_chest': 0.03, 'amp_ankle': 0.02, 'gyro_scale': 0.03},
    'standing':    {'freq': 0.2,  'amp_hand': 0.08, 'amp_chest': 0.03, 'amp_ankle': 0.02, 'gyro_scale': 0.02},
    'walking':     {'freq': 1.9,  'amp_hand': 0.80, 'amp_chest': 0.60, 'amp_ankle': 2.50, 'gyro_scale': 0.40},
    'running':     {'freq': 2.8,  'amp_hand': 1.50, 'amp_chest': 1.20, 'amp_ankle': 5.00, 'gyro_scale': 0.90},
    'cycling':     {'freq': 1.5,  'amp_hand': 0.40, 'amp_chest': 0.30, 'amp_ankle': 1.20, 'gyro_scale': 0.30},
    'nordic_walk': {'freq': 1.7,  'amp_hand': 2.20, 'amp_chest': 0.70, 'amp_ankle': 2.80, 'gyro_scale': 0.60},
    'stairs_up':   {'freq': 1.7,  'amp_hand': 1.00, 'amp_chest': 0.90, 'amp_ankle': 3.50, 'gyro_scale': 0.50},
    'stairs_down': {'freq': 1.6,  'amp_hand': 0.90, 'amp_chest': 0.80, 'amp_ankle': 3.20, 'gyro_scale': 0.45},
    'vacuuming':   {'freq': 1.0,  'amp_hand': 2.00, 'amp_chest': 0.40, 'amp_ankle': 0.80, 'gyro_scale': 0.55},
    'ironing':     {'freq': 0.8,  'amp_hand': 1.80, 'amp_chest': 0.15, 'amp_ankle': 0.10, 'gyro_scale': 0.50},
}

def synthesise_window(sensor, acc_x, acc_y, acc_z, intensity, rng, activity_name='walking'):
    """
    Build a realistic 100-row sensor window.
    - The baseline signal uses activity-specific frequency + amplitude profiles.
    - The user's slider values (acc_x, acc_y, acc_z) act as a DEVIATION from the
      expected baseline — simulating sensor error, misplacement, or drift.
    - intensity scales how energetic the user claims the motion is.
    - The sensor under test gets: baseline + user deviation + noise.
    - The other two sensors get: correct activity baseline + noise only.
      This models the realistic scenario where only one sensor is suspect.
    """
    N    = WINDOW_SIZE
    t    = np.linspace(0, 1, N)

    profile  = ACTIVITY_PROFILES.get(activity_name, ACTIVITY_PROFILES['walking'])
    freq     = profile['freq']
    gs       = profile['gyro_scale']

    # User deviation = difference between slider value and expected resting gravity
    # For a perfect sensor: acc_x≈0, acc_y≈0, acc_z≈9.81 at rest → deviation≈0
    # For a faulty sensor:  large deviation from these expected values
    dev_x = acc_x - 0.0
    dev_y = acc_y - 0.0
    dev_z = acc_z - 9.81

    def sensor_sig(mean_val, dev, amp, phase=0.0):
        """Activity baseline + user deviation + noise."""
        noise = 0.05 + float(intensity) * 0.01
        return (mean_val + dev
                + amp * np.sin(2 * np.pi * freq * t + phase)
                + rng.normal(0, noise, N))

    def correct_sig(mean_val, amp, phase=0.0):
        """Perfect baseline for sensors NOT under test."""
        noise = 0.04
        return mean_val + amp * np.sin(2 * np.pi * freq * t + phase) + rng.normal(0, noise, N)

    def gyro_sig(amp_scale, dev=0.0, phase=0.0):
        noise = 0.01
        return dev + amp_scale * gs * np.sin(2 * np.pi * freq * t + phase) + rng.normal(0, noise, N)

    def mag_sig(mean_val):
        return rng.normal(float(mean_val), 0.5, N)

    row = {}

    # Hand sensor
    ha = profile['amp_hand']
    if sensor == 'hand':
        row['hand_acc16_x'] = sensor_sig(0.0,  dev_x, ha * 0.6, 0.0)
        row['hand_acc16_y'] = sensor_sig(0.0,  dev_y, ha * 0.8, 0.5)
        row['hand_acc16_z'] = sensor_sig(9.81, dev_z, ha * 0.4, 1.0)
        row['hand_gyro_x']  = gyro_sig(1.0, dev_x * 0.05, 0.2)
        row['hand_gyro_y']  = gyro_sig(1.2, dev_y * 0.05, 0.7)
        row['hand_gyro_z']  = gyro_sig(0.8, dev_z * 0.03, 1.2)
    else:
        row['hand_acc16_x'] = correct_sig(0.0,  ha * 0.6, 0.0)
        row['hand_acc16_y'] = correct_sig(0.0,  ha * 0.8, 0.5)
        row['hand_acc16_z'] = correct_sig(9.81, ha * 0.4, 1.0)
        row['hand_gyro_x']  = gyro_sig(1.0, 0.0, 0.2)
        row['hand_gyro_y']  = gyro_sig(1.2, 0.0, 0.7)
        row['hand_gyro_z']  = gyro_sig(0.8, 0.0, 1.2)
    row['hand_mag_x'] = mag_sig(20.0)
    row['hand_mag_y'] = mag_sig(-5.0)
    row['hand_mag_z'] = mag_sig(42.0)

    # Chest sensor
    ca = profile['amp_chest']
    if sensor == 'chest':
        row['chest_acc16_x'] = sensor_sig(0.0,  dev_x, ca * 0.5, 0.3)
        row['chest_acc16_y'] = sensor_sig(0.0,  dev_y, ca * 0.6, 0.8)
        row['chest_acc16_z'] = sensor_sig(9.81, dev_z, ca * 0.3, 1.3)
        row['chest_gyro_x']  = gyro_sig(0.8, dev_x * 0.04, 0.4)
        row['chest_gyro_y']  = gyro_sig(0.9, dev_y * 0.04, 0.9)
        row['chest_gyro_z']  = gyro_sig(0.6, dev_z * 0.02, 1.4)
    else:
        row['chest_acc16_x'] = correct_sig(0.0,  ca * 0.5, 0.3)
        row['chest_acc16_y'] = correct_sig(0.0,  ca * 0.6, 0.8)
        row['chest_acc16_z'] = correct_sig(9.81, ca * 0.3, 1.3)
        row['chest_gyro_x']  = gyro_sig(0.8, 0.0, 0.4)
        row['chest_gyro_y']  = gyro_sig(0.9, 0.0, 0.9)
        row['chest_gyro_z']  = gyro_sig(0.6, 0.0, 1.4)
    row['chest_mag_x'] = mag_sig(20.0)
    row['chest_mag_y'] = mag_sig(-5.0)
    row['chest_mag_z'] = mag_sig(42.0)

    # Ankle sensor
    aa = profile['amp_ankle']
    if sensor == 'ankle':
        row['ankle_acc16_x'] = sensor_sig(0.0,  dev_x, aa * 0.9, 0.1)
        row['ankle_acc16_y'] = sensor_sig(0.0,  dev_y, aa * 1.0, 0.6)
        row['ankle_acc16_z'] = sensor_sig(9.81, dev_z, aa * 0.7, 1.1)
        row['ankle_gyro_x']  = gyro_sig(1.3, dev_x * 0.06, 0.3)
        row['ankle_gyro_y']  = gyro_sig(1.5, dev_y * 0.06, 0.8)
        row['ankle_gyro_z']  = gyro_sig(1.0, dev_z * 0.04, 1.3)
    else:
        row['ankle_acc16_x'] = correct_sig(0.0,  aa * 0.9, 0.1)
        row['ankle_acc16_y'] = correct_sig(0.0,  aa * 1.0, 0.6)
        row['ankle_acc16_z'] = correct_sig(9.81, aa * 0.7, 1.1)
        row['ankle_gyro_x']  = gyro_sig(1.3, 0.0, 0.3)
        row['ankle_gyro_y']  = gyro_sig(1.5, 0.0, 0.8)
        row['ankle_gyro_z']  = gyro_sig(1.0, 0.0, 1.3)
    row['ankle_mag_x'] = mag_sig(20.0)
    row['ankle_mag_y'] = mag_sig(-5.0)
    row['ankle_mag_z'] = mag_sig(42.0)

    return pd.DataFrame(row)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    """Health check — confirms model is loaded and shows class list."""
    if model is None:
        return jsonify({"loaded": False, "message": "Model not loaded"})
    # Convert numpy int64 to plain int for JSON
    classes = [int(c) for c in model.classes_]
    activities = [ACTIVITY_MAP.get(c, str(c)) for c in classes]
    return jsonify({
        "loaded": True,
        "model_type": type(model).__name__,
        "n_classes": len(classes),
        "activities": activities,
        "n_features": int(len(scaler.feature_names_in_)) if hasattr(scaler,'feature_names_in_') else None,
    })


@app.route("/api/check_sensor", methods=["POST"])
def check_sensor():
    """
    Sensor reliability check.
    Always returns JSON — never raises an unhandled exception.
    """
    # Always return JSON, even on crash
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model not loaded. Run pamap2_ml_model.py and add the save snippet."}), 503

        body = request.get_json(force=True, silent=True)
        if not body:
            return jsonify({"error": "Request body must be JSON"}), 400

        sensor        = str(body.get("sensor", "hand")).strip().lower()
        activity_name = str(body.get("activity", "walking")).strip().lower()
        acc_x     = float(body.get("acc_x",     0.0))
        acc_y     = float(body.get("acc_y",     0.0))
        acc_z     = float(body.get("acc_z",     9.81))
        intensity = float(body.get("intensity", 3.0))
        intensity = max(0.0, min(10.0, intensity))  # clamp 0-10

        if sensor not in ('hand', 'chest', 'ankle'):
            return jsonify({"error": f"sensor must be hand/chest/ankle, got: '{sensor}'"}), 400
        if activity_name not in ACTIVITY_NAME_TO_ID:
            return jsonify({"error": f"Unknown activity: '{activity_name}'. Valid: {list(ACTIVITY_NAME_TO_ID.keys())}"}), 400

        target_id = ACTIVITY_NAME_TO_ID[activity_name]

        # FIX: model.classes_ is numpy int64, compare properly
        classes_as_int = [int(c) for c in model.classes_]
        if target_id not in classes_as_int:
            return jsonify({"error": f"Activity '{activity_name}' (id={target_id}) not in trained model. "
                                     f"Model classes: {classes_as_int}"}), 400
        target_idx = classes_as_int.index(target_id)

        # Run 5 synthesis trials, average probabilities for stability
        N_TRIALS  = 5
        proba_sum = None
        for seed in range(N_TRIALS):
            rng       = np.random.default_rng(seed)
            window_df = synthesise_window(sensor, acc_x, acc_y, acc_z, intensity, rng, activity_name)
            proba, classes = predict_window(window_df)
            proba_sum = proba.copy() if proba_sum is None else proba_sum + proba

        proba_avg   = proba_sum / N_TRIALS
        reliability = float(proba_avg[target_idx])
        tl          = traffic_light(reliability)
        pct         = round(reliability * 100, 1)

        # Top predicted activity
        top_idx      = int(np.argmax(proba_avg))
        top_act_id   = int(classes[top_idx])
        top_activity = ACTIVITY_MAP.get(top_act_id, str(top_act_id))

        # Full probability dict — all plain Python floats for JSON
        all_probs = {}
        for c, p in zip(classes, proba_avg):
            act_name = ACTIVITY_MAP.get(int(c), str(int(c)))
            all_probs[act_name] = round(float(p), 4)
        top3 = sorted(all_probs.items(), key=lambda x: -x[1])[:3]

        # Verdict and advice text
        sensor_label = sensor.capitalize()
        act_display  = activity_name.replace('_', ' ')
        top_display  = top_activity.replace('_', ' ')

        if tl == "GREEN":
            verdict = (f"Your {sensor_label} sensor readings match {act_display}. "
                       f"Model confidence: {pct}%. Sensor is working correctly.")
            advice  = "No action needed. This sensor is reliable for activity tracking."
        elif tl == "YELLOW":
            verdict = (f"Your {sensor_label} sensor shows partial match with {act_display} ({pct}% confidence). ")
            if top_activity != activity_name:
                verdict += f"Readings look more like '{top_display}' — possible placement issue."
            advice = ("Check sensor placement — may be loose or slightly off position. "
                      "Re-run after repositioning. If persists, sensor may be degrading.")
        else:
            verdict = (f"Your {sensor_label} sensor does NOT match {act_display} (only {pct}% confidence). ")
            if top_activity != activity_name:
                verdict += f"Readings resemble '{top_display}' instead — sensor is likely faulty or detached."
            advice = ("Do not trust readings from this sensor. "
                      "Check cable connection, sensor placement, and hardware. "
                      "Replace if problem persists.")

        return jsonify({
            "sensor":          sensor,
            "activity":        activity_name,
            "reliability":     round(reliability, 4),
            "reliability_pct": pct,
            "traffic_light":   tl,
            "top_predicted":   top_activity,
            "verdict":         verdict,
            "advice":          advice,
            "top3":            top3,
            "all_probs":       all_probs,
            "n_trials":        N_TRIALS,
        })

    except Exception as e:
        # Always return JSON so the browser gets valid JSON, not an HTML error page
        err_trace = traceback.format_exc()
        print("ERROR in /api/check_sensor:")
        print(err_trace)
        return jsonify({
            "error": str(e),
            "trace": err_trace,
            "hint":  "Check Flask terminal for full traceback"
        }), 500


@app.route("/api/dashboard")
def dashboard():
    out = {}
    try:
        if os.path.exists(RESULTS_PATH):
            df  = pd.read_csv(RESULTS_PATH)
            tl  = df['traffic_light'].value_counts().to_dict() if 'traffic_light' in df.columns else {}
            out['traffic_summary'] = {
                'GREEN':     int(tl.get('GREEN', 0)),
                'YELLOW':    int(tl.get('YELLOW', 0)),
                'RED':       int(tl.get('RED', 0)),
                'total':     int(len(df)),
                'accuracy':  round(float(df['is_correct'].mean()) * 100, 1) if 'is_correct' in df.columns else None,
                'mean_conf': round(float(df['confidence'].mean()), 3)       if 'confidence' in df.columns else None,
            }
            if 'true_activity' in df.columns and 'is_correct' in df.columns:
                act = df.groupby('true_activity').agg(
                    accuracy  =('is_correct', 'mean'),
                    mean_conf =('confidence', 'mean'),
                    n         =('is_correct', 'count')
                ).round(3)
                out['activity_stats'] = act.reset_index().to_dict(orient='records')
            if 'true_activity' in df.columns and 'pred_activity' in df.columns:
                errors   = df[df['is_correct'] == 0]
                misclass = (errors.groupby(['true_activity', 'pred_activity'])
                                  .size().sort_values(ascending=False).head(8))
                out['misclassifications'] = [
                    {'true': t, 'pred': p, 'count': int(c)}
                    for (t, p), c in misclass.items()
                ]
        if os.path.exists(MODEL_CMP_PATH):
            out['model_comparison'] = pd.read_csv(MODEL_CMP_PATH).to_dict(orient='records')
        if os.path.exists(LOSO_PATH):
            out['loso'] = pd.read_csv(LOSO_PATH).to_dict(orient='records')
    except Exception as e:
        out['error'] = str(e)
    return jsonify(out)


@app.route("/graphs/<path:filename>")
def serve_graph(filename):
    return send_from_directory(PARENT, filename)


if __name__ == "__main__":
    print("\nPAMAP2 Sensor Reliability Checker")
    print("=" * 40)
    load_artefacts()
    print("=" * 40)
    print("Open: http://localhost:5000")
    print("Check model status: http://localhost:5000/api/status\n")
    app.run(debug=True, port=5000)