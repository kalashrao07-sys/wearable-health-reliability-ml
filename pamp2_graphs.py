"""
PAMAP2 - Graph Generation (FIXED + EXTENDED)
ML Course Project: Wearable Sensor Reliability Detection

FIXES in this version:
  G01: Bar labels used fmt='%,.0f' which matplotlib treats as a literal string
       → Fixed by passing labels=[f'{v:,}' for v in values] manually
  G04: 3-color pie for 9 subjects, pie charts are weak for comparison
       → Replaced with a clean horizontal bar chart coloured by data amount
  G05: Raw signals were flat because gravity (~9.8 m/s²) dominated the y-axis
       and the start of recording was a rest period. Fixed by:
       (a) sampling from the MIDDLE of each activity segment
       (b) mean-subtracting each signal to show only the dynamic oscillation
       (c) plotting HAND sensor which has the most activity-specific variation
  G12: Removing the ankle sensor INCREASED accuracy (drop = negative), which
       broke the y-axis scale and annotation positions. Redesigned to show
       absolute accuracy-without-sensor as bars, with baseline marked.
  G15: Partial dependence code produced a blank graph (wrong data source).
       Replaced with a Traffic Light donut chart — the core project finding.

NEW graphs added:
  G16: Standing vs Ironing sensor comparison — WHY the confusion happens
  G17: Confidence calibration curve — does high confidence actually mean accuracy?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# ─── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size':   11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

COLORS = {
    'blue':   '#2563EB',
    'green':  '#16A34A',
    'red':    '#DC2626',
    'orange': '#EA580C',
    'purple': '#7C3AED',
    'teal':   '#0891B2',
    'gray':   '#6B7280',
    'high':   '#16A34A',
    'medium': '#F59E0B',
    'low':    '#DC2626',
}

def save(name):
    plt.tight_layout()
    plt.savefig(name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {name}")

print("Loading data files...")
combined    = pd.read_csv("pamap2_combined.csv")
reliability = pd.read_csv("pamap2_reliability_results.csv")
model_comp  = pd.read_csv("pamap2_model_comparison.csv")
loso        = pd.read_csv("pamap2_loso_results.csv")
sensor_imp  = pd.read_csv("pamap2_sensor_importance.csv")
noise       = pd.read_csv("pamap2_noise_results.csv")
feat_imp    = pd.read_csv("pamap2_feature_importance.csv", index_col=0)
conf_matrix = pd.read_csv("pamap2_confusion_matrix.csv", index_col=0)

# Ensure traffic light column exists
if 'traffic_light' not in reliability.columns:
    reliability['traffic_light'] = reliability['confidence'].apply(
        lambda c: 'GREEN' if c >= 0.75 else ('YELLOW' if c >= 0.45 else 'RED'))

print("All files loaded. Generating graphs...\n")

# ── GRAPH 01: Activity Distribution ────────────────────────────────────────────
# FIX: fmt='%,.0f' was being printed as a literal string.
#      Must use labels= kwarg with manually formatted strings.
act_counts = combined.groupby('activity_name')['activityID'].count().sort_values()
fig, ax = plt.subplots(figsize=(10, 7))

high_intensity = {'running', 'rope_jumping', 'soccer'}
mid_intensity  = {'walking', 'cycling', 'nordic_walk', 'stairs_up',
                  'stairs_down', 'vacuuming', 'house_clean'}
bar_colors = [
    COLORS['red'] if n in high_intensity else
    COLORS['orange'] if n in mid_intensity else
    COLORS['blue']
    for n in act_counts.index
]
bars = ax.barh(act_counts.index, act_counts.values, color=bar_colors, alpha=0.85)
ax.bar_label(bars, labels=[f'{int(v):,}' for v in act_counts.values],
             padding=5, fontsize=9)
ax.set_title("Activity Distribution — Sensor Readings per Activity")
ax.set_xlabel("Number of rows (100 Hz sensor readings)")
ax.set_xlim(0, act_counts.max() * 1.18)
patches = [
    mpatches.Patch(color=COLORS['red'],    label='High intensity'),
    mpatches.Patch(color=COLORS['orange'], label='Medium intensity'),
    mpatches.Patch(color=COLORS['blue'],   label='Low / sedentary'),
]
ax.legend(handles=patches, fontsize=9, loc='lower right')
save("graph_01_activity_distribution.png")

# ── GRAPH 02: Heart Rate by Activity ───────────────────────────────────────────
hr = combined.groupby('activity_name')['heartRate'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = [COLORS['red'] if v > 130 else COLORS['orange'] if v > 100 else COLORS['blue']
              for v in hr.values]
bars = ax.bar(hr.index, hr.values, color=bar_colors, alpha=0.85)
ax.bar_label(bars, labels=[f'{int(v)}' for v in hr.values], padding=3, fontsize=9)
ax.set_title("Mean Heart Rate by Activity")
ax.set_ylabel("Heart Rate (BPM)")
ax.tick_params(axis='x', rotation=45)
ax.axhline(y=100, color=COLORS['gray'], linestyle='--', alpha=0.6, label='100 BPM reference')
ax.legend(fontsize=9)
save("graph_02_heartrate_by_activity.png")

# ── GRAPH 03: Acceleration Magnitude by Activity ────────────────────────────────
combined['acc_mag'] = np.sqrt(combined['chest_acc16_x']**2 +
                               combined['chest_acc16_y']**2 +
                               combined['chest_acc16_z']**2)
acc = combined.groupby('activity_name')['acc_mag'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = [COLORS['red'] if v > 11.5 else COLORS['orange'] if v > 10.5 else COLORS['blue']
              for v in acc.values]
bars = ax.bar(acc.index, acc.values, color=bar_colors, alpha=0.85)
ax.bar_label(bars, labels=[f'{v:.1f}' for v in acc.values], padding=3, fontsize=9)
ax.set_title("Mean Chest Acceleration Magnitude by Activity")
ax.set_ylabel("Acceleration Magnitude (m/s²)")
ax.tick_params(axis='x', rotation=45)
save("graph_03_acceleration_by_activity.png")

# ── GRAPH 04: Subject Distribution ─────────────────────────────────────────────
# FIX: 3-colour pie for 9 subjects looks wrong. A horizontal bar is clearer.
subj_counts = combined.groupby('subject_id')['activityID'].count().sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
cmap = plt.cm.Set2(np.linspace(0, 0.9, len(subj_counts)))
bars = ax.barh([f'Subject {s}' for s in subj_counts.index],
               subj_counts.values, color=cmap, alpha=0.9)
ax.bar_label(bars, labels=[f'{int(v):,} rows  ({v/subj_counts.sum()*100:.1f}%)'
                            for v in subj_counts.values],
             padding=5, fontsize=9)
ax.set_title("Data Distribution by Subject (after preprocessing)")
ax.set_xlabel("Number of sensor readings")
ax.set_xlim(0, subj_counts.max() * 1.4)
ax.axvline(subj_counts.mean(), color=COLORS['gray'], linestyle='--',
           linewidth=1, label=f'Mean: {int(subj_counts.mean()):,}')
ax.legend(fontsize=9)
save("graph_04_subject_distribution.png")

# ── GRAPH 05: Raw Sensor Signal ─────────────────────────────────────────────────
# FIX: Previous version plotted raw values dominated by gravity (≈9.8 m/s²)
# → All signals appeared as flat horizontal lines near 10 m/s².
# Fix: (a) sample from the MIDDLE of each activity (not the start/rest phase)
#      (b) mean-subtract each signal to remove DC bias and show oscillation
#      (c) use the HAND sensor which shows the most activity-specific variation
activities_to_show = [
    (5,  'Running',    COLORS['red']),
    (4,  'Walking',    COLORS['blue']),
    (2,  'Sitting',    COLORS['green']),
    (12, 'Stairs Up',  COLORS['orange']),
]
fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
fig.suptitle("Raw Hand Accelerometer Signal — Dynamic Motion per Activity\n"
             "(mean-subtracted to highlight oscillation; each panel = 3 seconds)",
             fontsize=11, fontweight='bold')
n_rows = 300  # 3 seconds at 100 Hz

for ax, (act_id, act_name, color) in zip(axes, activities_to_show):
    subset = combined[combined['activityID'] == act_id].reset_index(drop=True)
    # Sample from the MIDDLE of the recording (avoids rest/calibration phase at start)
    mid = max(0, len(subset) // 2 - n_rows // 2)
    chunk = subset.iloc[mid: mid + n_rows]
    t = np.linspace(0, 3, len(chunk))

    # Mean-subtract to remove gravity bias and show only dynamic motion
    x_dyn = chunk['hand_acc16_x'].values - chunk['hand_acc16_x'].mean()
    y_dyn = chunk['hand_acc16_y'].values - chunk['hand_acc16_y'].mean()

    ax.plot(t, x_dyn, color=color, linewidth=0.9, label='X-axis')
    ax.plot(t, y_dyn, color=color, linewidth=0.9, alpha=0.55, linestyle='--', label='Y-axis')
    ax.set_ylabel("m/s²", fontsize=9)
    ax.set_title(act_name, fontsize=10, fontweight='bold', color=color, loc='left', pad=3)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_ylim(-20, 20)
    ax.axhline(0, color=COLORS['gray'], linewidth=0.5, alpha=0.4)

axes[-1].set_xlabel("Time (seconds)")
axes[0].legend(fontsize=8, loc='upper right')
save("graph_05_raw_sensor_signal.png")

# ── GRAPH 06: Model Comparison ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
model_comp_sorted = model_comp.sort_values('accuracy')
n_models = len(model_comp_sorted)
model_palette = [COLORS['low'], COLORS['medium'], COLORS['medium'],
                 COLORS['high'], COLORS['purple']][:n_models]
x = np.arange(n_models)
w = 0.35

# Auto-detect balanced accuracy column name
bal_col = next((c for c in model_comp.columns if 'balanced' in c.lower()), None)

bars1 = ax.barh(x - (w/2 if bal_col else 0),
                model_comp_sorted['accuracy'] * 100,
                w if bal_col else 0.6,
                color=model_palette, alpha=0.85, label='Accuracy')
for bar in bars1:
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.1f}%', va='center', fontsize=9, fontweight='bold')

if bal_col:
    bars2 = ax.barh(x + w/2, model_comp_sorted[bal_col] * 100,
                    w, color=model_palette, alpha=0.45, hatch='//',
                    label='Balanced Accuracy')
    for bar in bars2:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.1f}%', va='center', fontsize=9)

ax.set_yticks(x)
ax.set_yticklabels(model_comp_sorted['model'], fontsize=10)
ax.set_xlabel("Score (%)")
ax.set_xlim(0, 108)
ax.set_title("Model Comparison: Accuracy vs Balanced Accuracy")
ax.axvline(80, color=COLORS['gray'], linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend(fontsize=9)
save("graph_06_model_comparison.png")

# ── GRAPH 07: Confusion Matrix ──────────────────────────────────────────────────
import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues',
            vmin=0, vmax=1, ax=ax,
            annot_kws={'size': 8},
            linewidths=0.3, linecolor='white',
            cbar_kws={'label': 'Proportion (row-normalized)'})
ax.set_title("Confusion Matrix — Subject 106 Test Set (Normalized by True Label)")
ax.set_xlabel("Predicted Activity")
ax.set_ylabel("True Activity")
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)
save("graph_07_confusion_matrix.png")

# ── GRAPH 08: Reliability per Activity ──────────────────────────────────────────
rel_summary = reliability.groupby('true_activity').agg(
    accuracy=('is_correct', 'mean')
).sort_values('accuracy')

fig, ax = plt.subplots(figsize=(9, 6))
bar_colors = [
    COLORS['low']    if v < 0.75 else
    COLORS['medium'] if v < 0.90 else
    COLORS['high']
    for v in rel_summary['accuracy'].values
]
bars = ax.barh(rel_summary.index, rel_summary['accuracy'] * 100,
               color=bar_colors, alpha=0.9)
ax.bar_label(bars, labels=[f'{v*100:.1f}%' for v in rel_summary['accuracy'].values],
             padding=4, fontsize=10, fontweight='bold')
ax.set_title("Reliability Score per Activity\n(% windows correctly classified after smoothing)")
ax.set_xlabel("Accuracy / Reliability (%)")
ax.set_xlim(0, 112)
ax.axvline(90, color=COLORS['high'],   linestyle='--', linewidth=1.2,
           alpha=0.7, label='HIGH threshold (90%)')
ax.axvline(75, color=COLORS['medium'], linestyle='--', linewidth=1.2,
           alpha=0.7, label='MEDIUM threshold (75%)')
patches = [
    mpatches.Patch(color=COLORS['high'],   label='HIGH reliability (≥90%)'),
    mpatches.Patch(color=COLORS['medium'], label='MEDIUM reliability (75–90%)'),
    mpatches.Patch(color=COLORS['low'],    label='LOW reliability (<75%)'),
]
ax.legend(handles=patches, fontsize=9, loc='lower right')
save("graph_08_reliability_per_activity.png")

# ── GRAPH 09: Confidence Distribution ────────────────────────────────────────────
reliable     = reliability[reliability['is_correct'] == 1]['confidence']
inconsistent = reliability[reliability['is_correct'] == 0]['confidence']
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0.1, 1.0, 20)
ax.hist(reliable,     bins=bins, color=COLORS['green'], alpha=0.7,
        label=f'Reliable ({len(reliable):,} windows)')
ax.hist(inconsistent, bins=bins, color=COLORS['red'],   alpha=0.75,
        label=f'Inconsistent ({len(inconsistent):,} windows)')
ax.axvline(0.70, color='black', linestyle='--', linewidth=1.5, label='0.7 threshold')
ax.set_xlabel("Confidence Score (max class probability)")
ax.set_ylabel("Number of Windows")
ax.set_title("Model Confidence: Reliable vs Inconsistent Readings")
ax.legend(fontsize=9)
save("graph_09_confidence_distribution.png")

# ── GRAPH 10: Feature Importance ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
feat_colors = []
for name in feat_imp.index:
    if 'ankle' in name:   feat_colors.append(COLORS['red'])
    elif 'hand' in name:  feat_colors.append(COLORS['blue'])
    else:                 feat_colors.append(COLORS['teal'])
bars = ax.barh(feat_imp.index, feat_imp['importance'] * 100,
               color=feat_colors, alpha=0.85)
ax.set_title("Top 20 Most Important Features (Random Forest)")
ax.set_xlabel("Feature Importance (%)")
patches = [
    mpatches.Patch(color=COLORS['red'],  label='Ankle sensor'),
    mpatches.Patch(color=COLORS['blue'], label='Hand sensor'),
    mpatches.Patch(color=COLORS['teal'], label='Chest sensor'),
]
ax.legend(handles=patches, fontsize=9)
save("graph_10_feature_importance.png")

# ── GRAPH 11: LOSO Results ──────────────────────────────────────────────────────
avg = loso['accuracy'].mean()
fig, ax = plt.subplots(figsize=(7, 4))
bar_colors = [COLORS['green'] if v >= avg else COLORS['orange']
              for v in loso['accuracy'].values]
bars = ax.bar([f'{int(s)}' for s in loso['subject']],
              loso['accuracy'] * 100, color=bar_colors, alpha=0.9)
ax.bar_label(bars, labels=[f'{v*100:.1f}%' for v in loso['accuracy'].values],
             padding=4, fontsize=10, fontweight='bold')
ax.axhline(y=avg * 100, color='black', linestyle='--', linewidth=1.2,
           label=f'Average: {avg:.1%}')
ax.set_title("Leave-One-Subject-Out Cross-Validation\n"
             "(How well does the model generalise to new people?)")
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Subjects")
ax.set_ylim(0, 110)
ax.legend(fontsize=9)
save("graph_11_loso_results.png")

# ── GRAPH 12: Sensor Importance ─────────────────────────────────────────────────
# FIX: Previous version showed "accuracy drop" which is negative for ankle
# (removing ankle INCREASES accuracy). Negative bars broke the layout.
# Solution: show absolute accuracy-without-sensor bars with baseline marked.
base_acc = model_comp[model_comp['model'] == 'Random Forest']['accuracy'].values[0]

labels_s  = ['All sensors\n(baseline)'] + \
            [f'Without\n{s.capitalize()} sensor' for s in sensor_imp['sensor']]
acc_vals  = [base_acc * 100] + list(sensor_imp['accuracy_without'] * 100)

# Colour logic: green if better than baseline, red if worse
bar_clrs = [COLORS['blue']] + [
    COLORS['green'] if v > base_acc * 100 else COLORS['red']
    for v in acc_vals[1:]
]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(labels_s, acc_vals, color=bar_clrs, alpha=0.9)
ax.bar_label(bars, labels=[f'{v:.1f}%' for v in acc_vals],
             padding=4, fontsize=11, fontweight='bold')
ax.axhline(y=base_acc * 100, color=COLORS['gray'], linestyle='--',
           linewidth=1.2, alpha=0.7, label=f'Baseline: {base_acc:.1%}')
ax.set_title("Model Accuracy With Each Sensor Location Removed\n"
             "(green = removal helps, red = removal hurts)")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(min(acc_vals) * 0.97, max(acc_vals) * 1.04)
ax.legend(fontsize=9)
save("graph_12_sensor_importance.png")

# ── GRAPH 13: Noise Injection ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(len(noise)), noise['accuracy'] * 100,
        marker='o', markersize=8, linewidth=2, color=COLORS['blue'])
for i, row in noise.iterrows():
    ax.annotate(f"{row['accuracy']*100:.1f}%",
                (i, row['accuracy'] * 100),
                textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
baseline = noise[noise['noise_level'] == 0.0]['accuracy'].values[0] * 100
ax.axhline(y=baseline, color=COLORS['green'], linestyle='--', alpha=0.7,
           label=f'Baseline (no noise): {baseline:.1f}%')
ax.fill_between(range(len(noise)), noise['accuracy'] * 100, baseline,
                alpha=0.12, color=COLORS['red'], label='Accuracy loss due to noise')
ax.set_title("Hand Sensor Noise Injection — Simulating Sensor Failure")
ax.set_xlabel("Noise Level (relative to signal std)")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(min(noise['accuracy'] * 100) * 0.97, baseline * 1.02)
ax.legend(fontsize=9)
ax.set_xticks(range(len(noise)))
ax.set_xticklabels(['Baseline\n(no noise)', 'Small\n(×0.1)', 'Medium\n(×0.5)',
                    'Large\n(×1.0)', 'Severe\n(×2.0)'])
save("graph_13_noise_injection.png")

# ── GRAPH 14: Reliability Over Time ────────────────────────────────────────────
color_map = {'GREEN': COLORS['green'], 'YELLOW': COLORS['orange'], 'RED': COLORS['red']}
fig, axes = plt.subplots(2, 1, figsize=(13, 8))
fig.suptitle("Reliability Score Over Time — Confidence Across Activities",
             fontsize=12, fontweight='bold')

ax = axes[0]
x = np.arange(len(reliability))
for tl, color in color_map.items():
    mask = reliability['traffic_light'] == tl
    ax.scatter(x[mask], reliability.loc[mask, 'confidence'],
               c=color, s=4, alpha=0.6, label=tl)
ax.axhline(0.75, color=COLORS['green'], linestyle='--', linewidth=0.8, alpha=0.7)
ax.axhline(0.45, color=COLORS['red'],   linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_ylabel("Confidence Score")
ax.set_ylim(0, 1.05)
ax.legend(title="Traffic Light", fontsize=8, loc='lower right',
          markerscale=3, title_fontsize=8)
ax.set_title("Model Confidence per Window (each dot = 1 second of sensor data)", fontsize=10)

ax = axes[1]
activity_colors = {
    'lying': COLORS['blue'],  'sitting': COLORS['teal'],  'standing': COLORS['purple'],
    'walking': COLORS['orange'], 'running': COLORS['red'],  'cycling': COLORS['green'],
    'nordic_walk': '#8B5CF6', 'stairs_up': '#F59E0B',   'stairs_down': '#D97706',
    'vacuuming': '#6B7280',   'ironing': '#EC4899',
}
for act in reliability['true_activity'].unique():
    mask = reliability['true_activity'] == act
    ax.scatter(x[mask], [act] * mask.sum(),
               c=activity_colors.get(act, COLORS['gray']), s=4, alpha=0.7)
ax.set_xlabel("Window Index (each = 1 second of data)")
ax.set_ylabel("True Activity")
ax.tick_params(axis='y', labelsize=8)
ax.set_title("Corresponding Activity Label", fontsize=10)
save("graph_14_reliability_over_time.png")

# ── GRAPH 15: Traffic Light Donut — Core Project Finding ────────────────────────
# FIX: Replaces the broken partial dependence graph.
# This is the single most important output graph for the project presentation.
tl_counts = reliability['traffic_light'].value_counts()
green_n  = tl_counts.get('GREEN',  0)
yellow_n = tl_counts.get('YELLOW', 0)
red_n    = tl_counts.get('RED',    0)
total_n  = len(reliability)

fig, ax = plt.subplots(figsize=(7, 7))
sizes  = [green_n, yellow_n, red_n]
colors_tl = ['#16A34A', '#F59E0B', '#DC2626']
labels_tl  = [
    f'RELIABLE\n{green_n:,} windows\n({green_n/total_n:.1%})',
    f'UNCERTAIN\n{yellow_n:,} windows\n({yellow_n/total_n:.1%})',
    f'UNRELIABLE\n{red_n:,} windows\n({red_n/total_n:.1%})',
]
wedges, texts = ax.pie(
    sizes, colors=colors_tl, startangle=90,
    wedgeprops=dict(width=0.52, edgecolor='white', linewidth=3),
    labels=labels_tl, labeldistance=1.12
)
for t in texts:
    t.set_fontsize(10)
ax.text(0, 0, f'{green_n/total_n:.1%}\nreliable',
        ha='center', va='center', fontsize=17, fontweight='bold', color='#16A34A')
ax.set_title("Traffic Light Reliability System\n"
             "Only 77% of wearable readings can be fully trusted",
             fontsize=13, fontweight='bold')
save("graph_15_traffic_light_donut.png")

# ── GRAPH 16: Standing vs Ironing — Why the Confusion Happens ───────────────────
# NEW: This is the key scientific insight of the project.
# Standing and ironing look the same to basic accelerometers but differ in gyroscope.
# This graph directly motivates why you added gyro magnitude features.
standing_df = combined[combined['activity_name'] == 'standing']
ironing_df  = combined[combined['activity_name'] == 'ironing']

# Sample equal amounts for fairness
n_sample = min(3000, len(standing_df), len(ironing_df))
np.random.seed(42)
st = standing_df.sample(n_sample)
ir = ironing_df.sample(n_sample)

sensors = {
    'Hand Acc\n(X-axis)':        ('hand_acc16_x',  False),
    'Hand Gyro\n(X-axis)\n★ KEY': ('hand_gyro_x',   True),
    'Ankle Acc\n(X-axis)':       ('ankle_acc16_x', False),
    'Heart\nRate':               ('heartRate',     False),
}

fig, axes = plt.subplots(1, 4, figsize=(13, 5))
fig.suptitle("Standing vs Ironing — Why the Sensor Sees Similar Signals\n"
             "(★ = the feature that separates them — gyroscope wrist rotation)",
             fontsize=11, fontweight='bold')

for ax, (title, (col, is_key)) in zip(axes, sensors.items()):
    bp = ax.boxplot(
        [st[col].values, ir[col].values],
        labels=['Standing', 'Ironing'],
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='.', markersize=1, alpha=0.2),
        widths=0.5
    )
    bp['boxes'][0].set_facecolor('#93C5FD')   # light blue for standing
    bp['boxes'][1].set_facecolor('#FCA5A5')   # light red for ironing

    if is_key:
        ax.set_facecolor('#FFFDE7')
        ax.set_title(title, fontsize=10, fontweight='bold', color=COLORS['orange'])
        ax.spines['bottom'].set_color(COLORS['orange'])
        ax.spines['left'].set_color(COLORS['orange'])
    else:
        ax.set_title(title, fontsize=10, fontweight='bold')

    ax.tick_params(axis='x', labelsize=9)
    ax.set_ylabel("Sensor value", fontsize=8)

patches_leg = [
    mpatches.Patch(color='#93C5FD', label='Standing'),
    mpatches.Patch(color='#FCA5A5', label='Ironing'),
]
axes[-1].legend(handles=patches_leg, fontsize=9, loc='upper right')
save("graph_16_standing_vs_ironing.png")

# ── GRAPH 17: Confidence Calibration Curve ──────────────────────────────────────
# NEW: Does high model confidence actually mean high accuracy?
# A well-calibrated model should have accuracy ≈ confidence.
# If the model says "90% confident" → it should be correct ~90% of the time.
conf_bins = np.arange(0.05, 1.05, 0.10)
bin_centers, bin_acc, bin_sizes = [], [], []

for lo in conf_bins[:-1]:
    hi = lo + 0.10
    mask = (reliability['confidence'] >= lo) & (reliability['confidence'] < hi)
    if mask.sum() >= 15:
        bin_centers.append((lo + hi) / 2)
        bin_acc.append(reliability.loc[mask, 'is_correct'].mean())
        bin_sizes.append(mask.sum())

fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(bin_centers, bin_acc,
                     s=[sz / 5 for sz in bin_sizes],
                     c=bin_acc, cmap='RdYlGn', vmin=0.5, vmax=1.0,
                     zorder=5, edgecolors='white', linewidth=0.8)
ax.plot(bin_centers, bin_acc, '-o', color=COLORS['blue'],
        linewidth=2, markersize=6, alpha=0.7)
ax.plot([0, 1], [0, 1], '--', color=COLORS['gray'], alpha=0.5,
        linewidth=1.5, label='Perfect calibration (accuracy = confidence)')
ax.fill_between(bin_centers, bin_acc, bin_centers,
                alpha=0.12, color=COLORS['blue'])
ax.set_xlabel("Model Confidence Score (max class probability)")
ax.set_ylabel("Actual Accuracy in That Confidence Bin")
ax.set_title("Confidence Calibration — Is the Model's Confidence Trustworthy?\n"
             "(Dots above the diagonal = overconfident; below = underconfident)")
ax.set_xlim(0.1, 1.05)
ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=9)
plt.colorbar(scatter, ax=ax, label='Accuracy', shrink=0.8)
save("graph_17_confidence_calibration.png")

print("\n── All 17 graphs saved ──────────────────────────────────────────────────")
print("Graph → PPT slide mapping:")
print("  graph_01  →  EDA: Activity Distribution (colour-coded by intensity)")
print("  graph_02  →  EDA: Mean Heart Rate per Activity")
print("  graph_03  →  EDA: Mean Acceleration Magnitude per Activity")
print("  graph_04  →  EDA: Subject Data Distribution (bar chart, not pie)")
print("  graph_05  →  Domain: Raw Sensor Signal (fixed — dynamic motion visible)")
print("  graph_06  →  Models: Accuracy vs Balanced Accuracy")
print("  graph_07  →  Performance: Confusion Matrix")
print("  graph_08  →  Finding 1: Reliability per Activity")
print("  graph_09  →  Finding 2: Confidence Distribution (reliable vs errors)")
print("  graph_10  →  Feature Engineering: Top 20 Features")
print("  graph_11  →  Finding 3: LOSO Cross-Validation (generalisation)")
print("  graph_12  →  Finding 4: Sensor Importance (fixed layout)")
print("  graph_13  →  Finding 5: Noise Injection (sensor failure simulation)")
print("  graph_14  →  Reliability Over Time")
print("  graph_15  →  KEY FINDING: Traffic Light Donut (core project claim)")
print("  graph_16  →  NEW: Standing vs Ironing sensor comparison")
print("  graph_17  →  NEW: Confidence Calibration Curve")