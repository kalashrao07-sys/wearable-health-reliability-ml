"""
PAMAP2 Dataset - Preprocessing (Protocol + Optional)
ML Course Project: Wearable Sensor Reliability Detection

What this script does:
  1. Loads Protocol CSV for ALL 9 subjects
  2. Also loads Optional CSV for subjects 101, 105, 106, 108, 109 (if file exists)
  3. Assigns proper column names (54 raw columns)
  4. Drops: activityID=0 rows, invalid orientation cols, uncalibrated acc6 cols
  5. Forward-fills heartRate NaNs, drops remaining sensor dropout NaN rows
  6. Adds data_source column (protocol / optional) for tracking
  7. Combines everything into one clean pamap2_combined.csv

Key difference vs Protocol-only:
  - Optional data is noisier and more realistic (freestyle recordings)
  - Window purity filtering happens in pamap2_ml_model.py during feature engineering
  - data_source column lets you analyse Protocol vs Optional separately if needed
"""

import pandas as pd
import numpy as np
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = "."
OUTPUT_FILE = "pamap2_combined.csv"

# All 9 protocol subjects
PROTOCOL_SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 108, 109]

# Optional files only exist for these subjects (per PAMAP2 dataset structure)
# Optional folder files are named: subject101Optional.csv etc.
# Adjust filename pattern below if your files are named differently
OPTIONAL_SUBJECTS = [101, 105, 106, 108, 109]
# ───────────────────────────────────────────────────────────────────────────────

ALL_COLS = [
    'timestamp', 'activityID', 'heartRate',
    # Hand IMU
    'hand_temp',
    'hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z',
    'hand_acc6_x',  'hand_acc6_y',  'hand_acc6_z',
    'hand_gyro_x',  'hand_gyro_y',  'hand_gyro_z',
    'hand_mag_x',   'hand_mag_y',   'hand_mag_z',
    'hand_ori1',    'hand_ori2',    'hand_ori3',   'hand_ori4',
    # Chest IMU
    'chest_temp',
    'chest_acc16_x','chest_acc16_y','chest_acc16_z',
    'chest_acc6_x', 'chest_acc6_y', 'chest_acc6_z',
    'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
    'chest_mag_x',  'chest_mag_y',  'chest_mag_z',
    'chest_ori1',   'chest_ori2',   'chest_ori3',  'chest_ori4',
    # Ankle IMU
    'ankle_temp',
    'ankle_acc16_x','ankle_acc16_y','ankle_acc16_z',
    'ankle_acc6_x', 'ankle_acc6_y', 'ankle_acc6_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
    'ankle_mag_x',  'ankle_mag_y',  'ankle_mag_z',
    'ankle_ori1',   'ankle_ori2',   'ankle_ori3',  'ankle_ori4'
]

DROP_COLS = [c for c in ALL_COLS if 'ori' in c or 'acc6' in c]

ACTIVITY_MAP = {
    1:'lying',        2:'sitting',       3:'standing',
    4:'walking',      5:'running',       6:'cycling',
    7:'nordic_walk',  9:'watching_TV',   10:'computer_work',
    11:'car_driving', 12:'stairs_up',    13:'stairs_down',
    16:'vacuuming',   17:'ironing',      18:'folding_laundry',
    19:'house_clean', 20:'soccer',       24:'rope_jumping'
}

# ─── LOAD FUNCTION ─────────────────────────────────────────────────────────────
def load_file(path: str, subject_id: int, source: str) -> pd.DataFrame:
    """
    Load one CSV file (Protocol or Optional), clean it, return DataFrame.
    source = 'protocol' or 'optional'
    """
    print(f"  [{source}] Loading {os.path.basename(path)}...")
    df = pd.read_csv(path, header=None, names=ALL_COLS)
    print(f"    Raw shape: {df.shape}")

    # Drop activityID=0 (transient noise — per README)
    before = len(df)
    df = df[df['activityID'] != 0]
    print(f"    After dropping activityID=0: {len(df):,} rows (removed {before-len(df):,})")

    # Drop invalid/uncalibrated columns
    df = df.drop(columns=DROP_COLS)

    # Forward-fill heartRate (9Hz vs 100Hz → 91% NaN is expected)
    df['heartRate'] = df['heartRate'].ffill()

    # Drop remaining NaN (sensor dropouts <0.5%)
    before = len(df)
    df = df.dropna()
    print(f"    After NaN handling: {len(df):,} rows")

    
   # Add tracking columns
    df['subject_id']   = subject_id
    df['data_source']  = source
    df['activity_name'] = df['activityID'].map(ACTIVITY_MAP)

    return df

# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────────
all_dfs = []
protocol_rows = 0
optional_rows = 0

print("=" * 60)
print("LOADING PROTOCOL FILES")
print("=" * 60)

for sid in PROTOCOL_SUBJECTS:
    # Try both .csv and .dat naming conventions
    for fname in [f"subject{sid}.csv", f"subject{sid}.dat"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df_s = load_file(path, sid, 'protocol')
            protocol_rows += len(df_s)
            all_dfs.append(df_s)
            break
    else:
        print(f"  WARNING: Protocol file for subject{sid} not found — skipping.")

print(f"\n{'='*60}")
print("LOADING OPTIONAL FILES")
print("=" * 60)

optional_loaded = 0
for sid in OPTIONAL_SUBJECTS:
    # Optional files are typically named subjectXXXOptional.csv
    found = False
    for fname in [f"subject{sid}Optional.csv",
                  f"subject{sid}_optional.csv",
                  f"subject{sid}optional.csv"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df_s = load_file(path, sid, 'optional')
            optional_rows += len(df_s)
            all_dfs.append(df_s)
            optional_loaded += 1
            found = True
            break
    if not found:
        print(f"  Optional file for subject{sid} not found — skipping.")
        print(f"  (Expected: subject{sid}Optional.csv)")

if optional_loaded == 0:
    print("\n  NOTE: No Optional files found. Running Protocol-only.")
    print("  If you have Optional files, rename them as: subject101Optional.csv")

# ─── COMBINE ───────────────────────────────────────────────────────────────────
combined = pd.concat(all_dfs, ignore_index=True)

print(f"\n{'='*60}")
print("COMBINED DATASET")
print(f"{'='*60}")
print(f"  Total shape  : {combined.shape}")
print(f"  Protocol rows: {protocol_rows:,}")
print(f"  Optional rows: {optional_rows:,}")
print(f"  Memory       : {combined.memory_usage(deep=True).sum()/1024/1024:.1f} MB")

print(f"\n── Data Source Split ───────────────────────────────────────")
print(combined['data_source'].value_counts().to_string())

print(f"\n── Activity Distribution (combined) ────────────────────────")
act_counts = combined.groupby('activity_name')['activityID'].count().sort_values(ascending=False)
print(act_counts.to_string())

print(f"\n── NaN Check ───────────────────────────────────────────────")
nan_counts = combined.isnull().sum()
print(nan_counts[nan_counts > 0] if nan_counts.sum() > 0 else "  No NaN values ✓")

# ─── SAVE ──────────────────────────────────────────────────────────────────────
combined.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved → {OUTPUT_FILE}")
print("\nNext: python pamap2_ml_model.py")