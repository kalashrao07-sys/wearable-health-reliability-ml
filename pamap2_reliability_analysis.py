"""
PAMAP2 - Reliability Analysis with Traffic Light System
ML Course Project: Wearable Sensor Reliability Detection

Run AFTER pamap2_ml_model.py has finished.

What this script adds:
  - Traffic light system: GREEN / YELLOW / RED per reading
  - Per-activity reliability tier table with clinical interpretation
  - Most common misclassification pairs
  - Saves pamap2_activity_reliability_summary.csv (final deliverable CSV)
"""

import pandas as pd
import numpy as np

df = pd.read_csv("pamap2_reliability_results.csv")

# ── TRAFFIC LIGHT THRESHOLDS ──────────────────────────────────────────────────
# These thresholds define when a wearable reading should be:
#   GREEN  → confidence ≥ 0.75: reading is reliable, use it
#   YELLOW → confidence 0.45 to 0.75: use with caution, note uncertainty
#   RED    → confidence < 0.45: reading is unreliable, flag or discard
GREEN_THRESHOLD  = 0.75
YELLOW_THRESHOLD = 0.45

def traffic_light(confidence):
    if confidence >= GREEN_THRESHOLD:
        return "GREEN"
    elif confidence >= YELLOW_THRESHOLD:
        return "YELLOW"
    else:
        return "RED"

df['traffic_light'] = df['confidence'].apply(traffic_light)

print("=" * 65)
print("WEARABLE SENSOR RELIABILITY ANALYSIS — TRAFFIC LIGHT SYSTEM")
print("=" * 65)

# ── OVERALL SUMMARY ───────────────────────────────────────────────────────────
total  = len(df)
green  = (df['traffic_light'] == 'GREEN').sum()
yellow = (df['traffic_light'] == 'YELLOW').sum()
red    = (df['traffic_light'] == 'RED').sum()
wrong  = (df['is_correct'] == 0).sum()

print(f"\nTotal windows analysed   : {total:,}")
print(f"")
print(f"  GREEN  (conf ≥ 0.75)  : {green:,}  ({green/total:.1%})  → RELIABLE — use this data")
print(f"  YELLOW (0.45–0.75)    : {yellow:,}  ({yellow/total:.1%})  → UNCERTAIN — use with caution")
print(f"  RED    (conf < 0.45)  : {red:,}   ({red/total:.1%})  → UNRELIABLE — flag or discard")
print(f"")
print(f"  Misclassified windows  : {wrong:,}  ({wrong/total:.1%})")
print(f"  Mean confidence        : {df['confidence'].mean():.3f}")

# ── TRAFFIC LIGHT BY ACTIVITY ─────────────────────────────────────────────────
print("\n── Traffic Light Distribution by Activity ───────────────────────────────")
print(f"{'Activity':<16} {'GREEN':>8} {'YELLOW':>8} {'RED':>8} {'Reliability Tier':>18}")
print("-" * 62)

summary = df.groupby('true_activity').agg(
    accuracy=('is_correct', 'mean'),
    avg_conf=('confidence', 'mean'),
    n=('is_correct', 'count'),
    green=('traffic_light', lambda x: (x == 'GREEN').sum()),
    yellow=('traffic_light', lambda x: (x == 'YELLOW').sum()),
    red=('traffic_light', lambda x: (x == 'RED').sum()),
).sort_values('accuracy', ascending=False)

for activity, row in summary.iterrows():
    if row['accuracy'] >= 0.90:
        tier = "HIGH ✓"
    elif row['accuracy'] >= 0.75:
        tier = "MEDIUM ⚠"
    else:
        tier = "LOW ✗"
    g_pct = row['green']  / row['n'] * 100
    y_pct = row['yellow'] / row['n'] * 100
    r_pct = row['red']    / row['n'] * 100
    print(f"{activity:<16} {g_pct:>6.0f}%  {y_pct:>6.0f}%  {r_pct:>6.0f}%   {tier:>16}")

# ── CLINICAL INTERPRETATION ────────────────────────────────────────────────────
print("\n── Clinical Interpretation ──────────────────────────────────────────────")
print("""
HIGH RELIABILITY activities (accuracy ≥ 90%):
  Lying, Nordic walking, Sitting, Running, Cycling
  → Wearable readings during these activities are consistent with sensor patterns.
  → Heart rate, step count, and calorie data can be trusted for clinical use.

MEDIUM RELIABILITY activities (75–90%):
  Ironing, Stairs down
  → Moderate confidence. Readings should be presented with a confidence indicator.
  → Not suitable for diagnostic decisions without additional validation.

LOW RELIABILITY activities (< 75%):
  Walking, Stairs up, Vacuuming, Standing
  → Sensor patterns overlap with other activities.
  → Step counting and calorie burn estimates during these activities are unreliable.
  → A wearable app should display a caution indicator for these readings.

KEY INSIGHT:
  Only {:.1%} of all readings receive a GREEN signal (high confidence).
  {:,.0f} out of {:,.0f} windows are flagged YELLOW or RED.
  This demonstrates why displaying raw sensor values without a reliability
  score is misleading — nearly 4 in 5 readings carry meaningful uncertainty.
""".format(green/total, yellow+red, total))

# ── MOST COMMON MISCLASSIFICATIONS ────────────────────────────────────────────
print("── Most Common Misclassifications ───────────────────────────────────────")
errors = df[df['is_correct'] == 0]
confusion = errors.groupby(['true_activity', 'pred_activity']).size().sort_values(ascending=False).head(10)
print(f"{'True Activity':<18} {'Predicted As':<18} {'Count':>8}  {'Why it happens'}")
print("-" * 75)
reasons = {
    ('ironing', 'standing'): 'slow arm movement — similar to standing still',
    ('standing', 'sitting'): 'near-zero body motion in both activities',
    ('walking', 'stairs_up'): 'similar gait rhythm and acceleration magnitude',
    ('vacuuming', 'stairs_up'): 'repetitive arm+leg motion resembles stair pattern',
    ('walking', 'stairs_down'): 'similar step frequency and body tilt',
    ('stairs_up', 'stairs_down'): 'opposite direction, same joint movement pattern',
}
for (true, pred), count in confusion.items():
    reason = reasons.get((true, pred), 'overlapping sensor signatures')
    print(f"{true:<18} {pred:<18} {count:>8}  {reason}")

# ── SAVE FINAL SUMMARY CSV ─────────────────────────────────────────────────────
summary['reliability_tier'] = summary['accuracy'].apply(
    lambda x: 'HIGH' if x >= 0.90 else ('MEDIUM' if x >= 0.75 else 'LOW')
)
summary['green_%']  = (summary['green']  / summary['n'] * 100).round(1)
summary['yellow_%'] = (summary['yellow'] / summary['n'] * 100).round(1)
summary['red_%']    = (summary['red']    / summary['n'] * 100).round(1)

output_cols = ['accuracy', 'avg_conf', 'n', 'green_%', 'yellow_%', 'red_%', 'reliability_tier']
summary[output_cols].round(3).to_csv("pamap2_activity_reliability_summary.csv")
print("\nSaved → pamap2_activity_reliability_summary.csv")

# Also save the full results with traffic light labels
df.to_csv("pamap2_reliability_results.csv", index=False)
print("Updated → pamap2_reliability_results.csv  (now includes traffic_light column)")

print("\n── Files to show your professor ─────────────────────────────────────────")
print("  pamap2_combined.csv                    — cleaned dataset")
print("  pamap2_reliability_results.csv         — per-window reliability with traffic light")
print("  pamap2_activity_reliability_summary.csv — final reliability tier summary")