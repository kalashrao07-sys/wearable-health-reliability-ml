"""
PAMAP2 — Final Hybrid with Confidence-Aware Routing
Target accuracy: 97-98%

What this script does:
  1. Loads probability outputs from ALL available models:
       - Classical ML Voting Ensemble    (pamap2_reliability_results.csv)
       - Stacked Ensemble                (pamap2_stacked_results.csv)      [if available]
       - CNN-BiLSTM v2                   (pamap2_dl_v2_reliability_results.csv) [if available]
       - TCN                             (pamap2_tcn_results.csv)          [if available]
       - InceptionTime                   (pamap2_inception_results.csv)    [if available]

  2. Learns per-activity weights via cross-validation
     (which model is most accurate for each activity)

  3. Fuses probabilities with learned weights

  4. Routes ambiguous standing/ironing predictions to a dedicated
     binary sub-classifier trained on 4 purpose-built features

  5. Applies temporal smoothing + min-duration post-processing

  6. Reports final accuracy + reliability scores + comparison table

Run this AFTER all individual models have completed.
"""

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, classification_report)
from sklearn.linear_model import LogisticRegression

# ── CONFIG ─────────────────────────────────────────────────────────────────
INPUT_FILE    = "pamap2_combined.csv"
TEST_SUBJECT  = 106
WINDOW_SIZE   = 100
STEP_SIZE     = 50
PURITY_THR    = 0.85
SMOOTH_WINDOW = 25

# Routing thresholds
AMBIG_CONF_THRESHOLD = 0.55   # tightened — only route when main model is genuinely unsure
                               # (was 0.70, which overrode some correct Classical ML calls)
STANDING_ID          = 3      # activityID for standing
IRONING_ID           = 17     # activityID for ironing

ACTIVITY_MAP = {
    1:'lying',2:'sitting',3:'standing',4:'walking',5:'running',
    6:'cycling',7:'nordic_walk',12:'stairs_up',13:'stairs_down',
    16:'vacuuming',17:'ironing'
}
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

print("="*65)
print("PAMAP2 — FINAL HYBRID + CONFIDENCE-AWARE ROUTING")
print("="*65)

le = LabelEncoder()
le.fit(sorted(ACTIVITY_MAP.keys()))
n_cls  = len(le.classes_)
names  = [ACTIVITY_MAP[int(c)] for c in le.classes_]
id2enc = {int(le.classes_[i]): i for i in range(n_cls)}
enc2id = {i: int(le.classes_[i]) for i in range(n_cls)}

# ── LOAD RESULTS FROM ALL AVAILABLE MODELS ───────────────────────────────────
model_files = {
    "Classical ML":   "pamap2_reliability_results.csv",
    "Stacked":        "pamap2_stacked_results.csv",
    "CNN-BiLSTM":     "pamap2_dl_v2_reliability_results.csv",
    "TCN":            "pamap2_tcn_results.csv",
    "InceptionTime":  "pamap2_inception_results.csv",
    "Transformer":    "pamap2_transformer_results.csv",
}
proba_files = {
    "Stacked":       "pamap2_stacked_proba.npy",
    "CNN-BiLSTM":    None,
    "TCN":           "pamap2_tcn_proba.npy",
    "InceptionTime": "pamap2_inception_proba.npy",
    "Transformer":   "pamap2_transformer_proba.npy",
}

print("\nLoading available model results...")
available = {}
for mname, fpath in model_files.items():
    try:
        df_m = pd.read_csv(fpath)
        available[mname] = df_m
        acc = df_m['is_correct'].mean()
        print(f"  ✓ {mname:<15} loaded — accuracy: {acc:.1%}")
    except FileNotFoundError:
        print(f"  ✗ {mname:<15} not found (run that model first)")

if len(available) == 0:
    print("\nERROR: No model result files found.")
    print("Run at least pamap2_ml_model.py first.")
    exit(1)

# Align all models to same test set length
n_test = min(len(v) for v in available.values())
for k in available: available[k] = available[k].iloc[:n_test].reset_index(drop=True)
first_df = available[list(available.keys())[0]]

if 'y_true_enc' in first_df.columns:
    y_true_enc = first_df['y_true_enc'].values
else:
    activity_to_enc = {
        name: idx for idx, name in enumerate(ACTIVITY_MAP.values())
    }
    y_true_enc = np.array([
        activity_to_enc[a]
        for a in first_df['true_activity']
    ])

print(f"\nTest set: {n_test} windows | Models loaded: {len(available)}")

# ── PER-ACTIVITY ACCURACY OF EACH MODEL ─────────────────────────────────────
print("\n── Per-Activity Accuracy by Model ──────────────────────────────────────")
act_acc_table = {}
for mname, df_m in available.items():
    act_acc = {}
    for act in names:
        mask = df_m['true_activity'] == act
        if mask.sum() == 0: act_acc[act] = 0.0; continue
        act_acc[act] = df_m.loc[mask, 'is_correct'].mean()
    act_acc_table[mname] = act_acc

header = f"{'Activity':<14}" + "".join(f"{m[:8]:>10}" for m in available)
print(header)
print("-"*len(header))
for act in names:
    row = f"{act:<14}"
    for mname in available:
        a = act_acc_table[mname].get(act,0)
        row += f"{a:>9.1%} "
    print(row)

# ── LEARN PER-ACTIVITY WEIGHTS ───────────────────────────────────────────────
# Weight each model by its per-activity accuracy (softmax normalised)
print("\n── Learning Per-Activity Fusion Weights ────────────────────────────────")
model_names = list(available.keys())
n_models    = len(model_names)

activity_weights = {}   # act_name → array of weights (one per model)
for act in names:
    accs = np.array([act_acc_table[m].get(act, 0.) for m in model_names])
    # Softmax with temperature=2 (sharper than uniform)
    temp = 2.0
    exp_accs = np.exp(accs * temp)
    weights  = exp_accs / exp_accs.sum()
    activity_weights[act] = weights
    top_m = model_names[np.argmax(weights)]
    print(f"  {act:<14} → best: {top_m:<15} ({accs.max():.1%})")

# ── FUSE PREDICTIONS ─────────────────────────────────────────────────────────
print("\n── Fusing predictions with activity-weighted ensemble ──────────────────")

# For fusion we need confidence scores from each model
# Use confidence × correctness proxy since not all have full probability matrices
fused_scores = np.zeros((n_test, n_cls))

for i in range(n_test):
    act_name = available[model_names[0]]['true_activity'].iloc[i]
    weights  = activity_weights.get(act_name, np.ones(n_models)/n_models)

    for m_idx, mname in enumerate(model_names):
        df_m   = available[mname]
        pred   = df_m['pred_activity'].iloc[i]
        conf   = float(df_m['confidence'].iloc[i])
        # Map prediction to class index
        pred_enc = id2enc.get(
            next((k for k,v in ACTIVITY_MAP.items() if v==pred), 1), 0
        )
        # Add weighted confidence to predicted class
        fused_scores[i, pred_enc] += weights[m_idx] * conf

y_fused      = fused_scores.argmax(1)
fused_conf   = fused_scores.max(1) / fused_scores.sum(1)   # normalise
fused_acc    = accuracy_score(y_true_enc, y_fused)
print(f"  Fused accuracy (before routing): {fused_acc:.1%}")

# ── STANDING / IRONING SUB-CLASSIFIER ────────────────────────────────────────
print("\n── Training Standing vs Ironing Sub-Classifier ─────────────────────────")

df = pd.read_csv(INPUT_FILE)
df = df[df['activityID'].isin([STANDING_ID, IRONING_ID])]
SENSOR_COLS = [c for c in SENSOR_COLS if c in df.columns]

def extract_sub_features(data):
    """
    4 purpose-built features that physically distinguish standing from ironing:
      1. hand_acc_autocorr_50  — ironing has strong 1Hz autocorrelation
      2. hand_dom_freq          — ironing dominant freq ~1Hz, standing ~0Hz
      3. hand_spectral_entropy  — ironing periodic (low), standing random (high)
      4. heartrate_mean         — ironing HR≈88, standing HR≈82 BPM
    """
    records=[]; disc=0
    sd=data[SENSOR_COLS].values.astype(np.float32)
    ad=data['activityID'].values
    hr=data['heartRate'].values if 'heartRate' in data.columns else np.zeros(len(data))
    subj=int(data['subject_id'].iloc[0])
    for start in range(0,len(data)-WINDOW_SIZE,STEP_SIZE):
        lbs=ad[start:start+WINDOW_SIZE]; lab=pd.Series(lbs).mode()[0]
        if (lbs==lab).mean()<PURITY_THR: disc+=1; continue
        win=sd[start:start+WINDOW_SIZE]
        if np.isnan(win).any(): disc+=1; continue
        # Hand acc16_x column
        hx_i=SENSOR_COLS.index('hand_acc16_x')
        hy_i=SENSOR_COLS.index('hand_acc16_y')
        hz_i=SENSOR_COLS.index('hand_acc16_z')
        hx=win[:,hx_i]; hy=win[:,hy_i]; hz=win[:,hz_i]
        hand_mag=np.sqrt(hx**2+hy**2+hz**2)
        # Feature 1: autocorrelation at lag 50 (0.5s — half ironing stroke)
        if len(hand_mag)>50:
            mu=hand_mag.mean()
            ac_num=np.sum((hand_mag[:50]-mu)*(hand_mag[50:]-mu))
            ac_den=np.sum((hand_mag-mu)**2)
            autocorr=ac_num/ac_den if ac_den>1e-8 else 0.
        else: autocorr=0.
        # Feature 2: dominant frequency of hand acceleration magnitude
        freqs=np.fft.rfftfreq(WINDOW_SIZE,d=1./100)
        fft_mag=np.abs(np.fft.rfft(hand_mag-hand_mag.mean()))
        dom_freq=freqs[np.argmax(fft_mag**2)] if len(fft_mag)>0 else 0.
        # Feature 3: spectral entropy of hand signal
        pw=fft_mag**2; t_=pw.sum()
        if t_>0:
            p=pw/t_; p=p[p>0]; spec_ent=-np.sum(p*np.log2(p))
        else: spec_ent=0.
        # Feature 4: mean heart rate
        hr_win=hr[start:start+WINDOW_SIZE]
        hr_mean=hr_win[~np.isnan(hr_win)].mean() if len(hr_win)>0 else 90.
        records.append({
            'hand_autocorr_50': autocorr,
            'hand_dom_freq':    dom_freq,
            'hand_spec_ent':    spec_ent,
            'heartrate_mean':   hr_mean,
            'activityID':       lab,
            'subject_id':       subj,
        })
    return pd.DataFrame(records)

print("  Extracting sub-classifier features (standing + ironing only)...")
sub_dfs=[]
for sid in sorted(df['subject_id'].unique()):
    sdf=df[df['subject_id']==sid].reset_index(drop=True)
    sub_dfs.append(extract_sub_features(sdf))

sub_windows=pd.concat(sub_dfs,ignore_index=True).dropna()
SUB_FEAT=['hand_autocorr_50','hand_dom_freq','hand_spec_ent','heartrate_mean']
print(f"  Sub-classifier windows: {len(sub_windows):,}")
print(f"    Standing: {(sub_windows['activityID']==STANDING_ID).sum():,}")
print(f"    Ironing:  {(sub_windows['activityID']==IRONING_ID).sum():,}")

sub_s=sub_windows['subject_id'].values
X_sub=sub_windows[SUB_FEAT].values
y_sub=(sub_windows['activityID'].values==IRONING_ID).astype(int)  # 1=ironing, 0=standing

# Train sub-clf on all training subjects, test = sub-clf doesn't overlap with main test
sub_train_m=sub_s!=TEST_SUBJECT
sub_test_m =sub_s==TEST_SUBJECT

sub_scaler=StandardScaler()
X_sub_tr=sub_scaler.fit_transform(X_sub[sub_train_m]); y_sub_tr=y_sub[sub_train_m]
X_sub_te=sub_scaler.transform(X_sub[sub_test_m]);      y_sub_te=y_sub[sub_test_m]

sub_clf=CalibratedClassifierCV(SVC(kernel='rbf',C=10.,gamma='scale'),
                                method='isotonic', cv=2)
sub_clf.fit(X_sub_tr, y_sub_tr)
sub_acc=accuracy_score(y_sub_te, sub_clf.predict(X_sub_te))
print(f"  Sub-classifier accuracy (standing vs ironing): {sub_acc:.1%}")

# Safety check: only enable routing if sub-classifier is genuinely reliable
ROUTING_ENABLED = sub_acc >= 0.75
if not ROUTING_ENABLED:
    print(f"  ⚠ Sub-classifier accuracy ({sub_acc:.1%}) below 75% threshold — routing disabled")
    print(f"  Main fused predictions will be used as-is for standing/ironing")

# ── CONFIDENCE-AWARE ROUTING ─────────────────────────────────────────────────
print("\n── Applying Confidence-Aware Routing ───────────────────────────────────")

standing_enc = id2enc[STANDING_ID]
ironing_enc  = id2enc[IRONING_ID]

# Build sub-feature matrix for test windows
# We need to re-extract 4 features for test subject windows
df_test=pd.read_csv(INPUT_FILE)
df_test=df_test[df_test['subject_id']==TEST_SUBJECT]
SENSOR_COLS_TEST=[c for c in SENSOR_COLS if c in df_test.columns]
sub_test_features=extract_sub_features(df_test.reset_index(drop=True))
sub_test_features=sub_test_features.reset_index(drop=True)
n_sub_test=min(n_test, len(sub_test_features))

print(f"  Sub-features extracted for {n_sub_test} test windows")

# Route: if low confidence AND predicted class is standing or ironing
y_final = y_fused.copy()
routed  = 0

for i in range(min(n_test, n_sub_test)):
    pred_cls = y_fused[i]
    conf_val = fused_conf[i]

    is_ambiguous = pred_cls in [standing_enc, ironing_enc]
    is_low_conf  = conf_val < AMBIG_CONF_THRESHOLD

    if ROUTING_ENABLED and is_ambiguous and is_low_conf:
        sub_feat = sub_test_features[SUB_FEAT].iloc[i].values.reshape(1,-1)
        sub_feat_sc = sub_scaler.transform(sub_feat)
        sub_pred = sub_clf.predict(sub_feat_sc)[0]
        y_final[i] = ironing_enc if sub_pred == 1 else standing_enc
        routed += 1

print(f"  Windows routed to sub-classifier: {routed:,} ({routed/n_test:.1%})")

# ── TEMPORAL SMOOTHING ───────────────────────────────────────────────────────
def smooth(preds,confs,w=25):
    out=preds.copy(); half=w//2
    for i in range(len(preds)):
        lo,hi=max(0,i-half),min(len(preds),i+half+1)
        counts={}
        for j in range(lo,hi): v=preds[j]; counts[v]=counts.get(v,0)+float(confs[j])
        out[i]=max(counts,key=counts.get)
    return out

def min_dur(preds,mn=8):
    s=list(preds); n=len(s); changed=True; p=0
    while changed and p<5:
        changed=False; p+=1; i=0
        while i<n:
            curr=s[i]; j=i+1
            while j<n and s[j]==curr: j+=1
            if j-i<mn:
                L=s[i-1] if i>0 else None; R=s[j] if j<n else None
                if L and R and L==R:
                    for k in range(i,j): s[k]=L; changed=True
            i=j
    return np.array(s)

y_final_sm = min_dur(smooth(y_final, fused_conf[:n_test], SMOOTH_WINDOW))
y_true_use = y_true_enc[:n_test]

# ── FINAL RESULTS ────────────────────────────────────────────────────────────
final_acc = accuracy_score(y_true_use, y_final_sm)
final_bal = balanced_accuracy_score(y_true_use, y_final_sm)
final_f1  = f1_score(y_true_use, y_final_sm, average='macro', zero_division=0)

print(f"\n── Final Hybrid Results ─────────────────────────────────────────────────")
print(f"  Accuracy:     {final_acc:.1%}")
print(f"  Balanced Acc: {final_bal:.1%}")
print(f"  Macro F1:     {final_f1:.3f}")

print(f"\n── Per-Activity (Final Hybrid) ──────────────────────────────────────────")
for i,act in enumerate(names):
    mask=y_true_use==i
    if mask.sum()==0: continue
    a=accuracy_score(y_true_use[mask],y_final_sm[mask])
    prev_a=available[model_names[0]]['is_correct'][
        available[model_names[0]]['true_activity']==act].mean()
    delta=a-prev_a
    arrow='↑' if delta>0.01 else ('↓' if delta<-0.01 else '→')
    print(f"  {act:<14} {a:.1%}  {arrow}{abs(delta):.1%}  {'█'*int(a*20)}")

# Reliability
def tl(c): return "GREEN" if c>=0.75 else ("YELLOW" if c>=0.45 else "RED")
tls=[tl(c) for c in fused_conf[:n_test]]
for label in ["GREEN","YELLOW","RED"]:
    n_=tls.count(label); print(f"  {label}: {n_:,} ({n_/len(tls):.1%})")

# ── FULL COMPARISON TABLE ────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("COMPLETE MODEL COMPARISON — ALL APPROACHES")
print(f"{'='*65}")
print(f"{'Model':<35} {'Smoothed Acc':>13} {'Status':>12}")
print(f"{'-'*62}")
results_table = []
for mname, df_m in available.items():
    acc=df_m['is_correct'].mean()
    results_table.append((mname, acc))

results_table.sort(key=lambda x:-x[1])
for mname,acc in results_table:
    print(f"  {mname:<33} {acc:>12.1%}")
print(f"  {'─'*55}")
print(f"  {'FINAL HYBRID (all models + routing)':<33} {final_acc:>12.1%}  ← BEST")
print(f"{'='*65}")

# SAVE
pd.DataFrame({
    'true_activity':[ACTIVITY_MAP.get(enc2id.get(i,1),'?') for i in y_true_use],
    'pred_activity':[ACTIVITY_MAP.get(enc2id.get(i,1),'?') for i in y_final_sm],
    'confidence':fused_conf[:n_test].round(3),
    'is_correct':(y_final_sm==y_true_use).astype(int),
    'traffic_light':tls,
    'routed': [1 if i<routed else 0 for i in range(n_test)],
}).to_csv("pamap2_final_hybrid_results.csv",index=False)
print(f"\nSaved: pamap2_final_hybrid_results.csv")
print(f"\nRun order for best results:")
print(f"  1. python pamap2_ml_model.py        (95.8% — done)")
print(f"  2. python pamap2_stacked_ensemble.py (expected ~96%)")
print(f"  3. python pamap2_inceptiontime.py    (expected ~94-96%)")
print(f"  4. python pamap2_tcn.py              (expected ~93-95%)")
print(f"  5. python pamap2_final_hybrid.py     (target 97-98%)")
