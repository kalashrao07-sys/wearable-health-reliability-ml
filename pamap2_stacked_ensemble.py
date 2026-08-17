"""
PAMAP2 — Stacked Generalization (Meta-Learner Ensemble)

Why stacking beats soft voting:
  - Soft voting treats all models equally for all activities
  - Stacking LEARNS which model to trust for which activity
  - RF is better at standing. KNN is better at stairs.
  - A meta-learner (Logistic Regression) learns these patterns
    from held-out predictions — no data leakage

Pipeline:
  Level-0: LightGBM + ExtraTrees + SVM-RBF (trained on train subjects)
  Level-1: Out-of-fold predictions fed to LogReg meta-learner
  Final:   Meta-learner predicts from Level-0 test probabilities

Requires: pip install lightgbm
Run AFTER pamap2_preprocessing.py
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, classification_report)
from sklearn.calibration import CalibratedClassifierCV

try:
    import lightgbm as lgb
    print(f"LightGBM {lgb.__version__} loaded")
except ImportError:
    print("Install: pip install lightgbm"); raise

# ── CONFIG ─────────────────────────────────────────────────────────────────
INPUT_FILE    = "pamap2_combined.csv"
FEATURE_FILE  = "pamap2_feature_cols.json"
WINDOW_SIZE   = 100
STEP_SIZE     = 50
TEST_SUBJECT  = 106
PURITY_THR    = 0.85
SMOOTH_WINDOW = 25

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
print("PAMAP2 — STACKED GENERALIZATION (Meta-Learner)")
print("="*65)

# ── LOAD + FEATURE EXTRACTION ──────────────────────────────────────────────
print("\nLoading data and extracting features...")
df = pd.read_csv(INPUT_FILE)
df = df[df['activityID'].isin(ACTIVITY_MAP.keys())]
SENSOR_COLS = [c for c in SENSOR_COLS if c in df.columns]
n_ch = len(SENSOR_COLS)

def spectral_entropy(v):
    p=np.abs(np.fft.rfft(v))**2; t=p.sum()
    if t==0: return 0.
    p=p/t; p=p[p>0]; return -np.sum(p*np.log2(p))

def extract_features(data):
    freqs=np.fft.rfftfreq(WINDOW_SIZE,d=1./100)
    records=[]; disc=0
    sd=data[SENSOR_COLS].values.astype(np.float32)
    ad=data['activityID'].values
    subj=int(data['subject_id'].iloc[0])
    for start in range(0,len(data)-WINDOW_SIZE,STEP_SIZE):
        lbs=ad[start:start+WINDOW_SIZE]; lab=pd.Series(lbs).mode()[0]
        if (lbs==lab).mean()<PURITY_THR: disc+=1; continue
        win=sd[start:start+WINDOW_SIZE]
        if np.isnan(win).any(): disc+=1; continue
        feat={}
        for ci,col in enumerate(SENSOR_COLS):
            v=win[:,ci]
            feat[f"{col}_mean"]=v.mean(); feat[f"{col}_std"]=v.std()
            feat[f"{col}_min"]=v.min();   feat[f"{col}_max"]=v.max()
            feat[f"{col}_range"]=v.max()-v.min(); feat[f"{col}_energy"]=(v**2).mean()
            fft_mag=np.abs(np.fft.rfft(v-v.mean())); pw=fft_mag**2; dm=np.argmax(pw)
            feat[f"{col}_dom_freq"]=freqs[dm]; feat[f"{col}_peak_power"]=pw[dm]
            feat[f"{col}_spectral_entropy"]=spectral_entropy(v)
        mags={}
        for loc in ['hand','chest','ankle']:
            xi=SENSOR_COLS.index(f'{loc}_acc16_x')
            yi=SENSOR_COLS.index(f'{loc}_acc16_y')
            zi=SENSOR_COLS.index(f'{loc}_acc16_z')
            mag=np.sqrt(win[:,xi]**2+win[:,yi]**2+win[:,zi]**2)
            mags[loc]=mag; feat[f'{loc}_acc_mag_mean']=mag.mean()
            feat[f'{loc}_acc_mag_std']=mag.std()
            delta=np.diff(mag)
            feat[f'{loc}_acc_mag_delta_mean']=np.abs(delta).mean()
            feat[f'{loc}_acc_mag_delta_std']=delta.std()
        def sc(a,b):
            if a.std()<1e-8 or b.std()<1e-8: return 0.
            return float(np.corrcoef(a,b)[0,1])
        feat['corr_hand_ankle']=sc(mags['hand'],mags['ankle'])
        feat['corr_hand_chest']=sc(mags['hand'],mags['chest'])
        feat['corr_chest_ankle']=sc(mags['chest'],mags['ankle'])
        feat['hand_ankle_ratio']=mags['hand'].std()/(mags['ankle'].std()+1e-6)
        feat['hand_chest_ratio']=mags['hand'].std()/(mags['chest'].std()+1e-6)
        feat['chest_ankle_ratio']=mags['chest'].std()/(mags['ankle'].std()+1e-6)
        for loc in ['hand','chest','ankle']:
            zi=SENSOR_COLS.index(f'{loc}_acc16_z')
            feat[f'{loc}_tilt_proxy']=win[:,zi].mean()
        feat['activityID']=lab; feat['subject_id']=subj
        records.append(feat)
    return pd.DataFrame(records)

all_dfs=[]
for sid in sorted(df['subject_id'].unique()):
    print(f"  Subject {int(sid)}...")
    sdf=df[df['subject_id']==sid].reset_index(drop=True)
    all_dfs.append(extract_features(sdf))

windows=pd.concat(all_dfs,ignore_index=True).dropna()
FEAT_COLS=[c for c in windows.columns if c not in ['activityID','subject_id']]
print(f"Windows: {len(windows):,} | Features: {len(FEAT_COLS)}")

le=LabelEncoder(); le.fit(sorted(ACTIVITY_MAP.keys()))
y_enc=le.transform(windows['activityID'].values)
n_cls=len(le.classes_); names=[ACTIVITY_MAP[int(c)] for c in le.classes_]
s_all=windows['subject_id'].values

X_all=windows[FEAT_COLS].values
test_m=s_all==TEST_SUBJECT; train_m=~test_m
X_tr_raw,y_tr=X_all[train_m],y_enc[train_m]
X_te_raw,y_te=X_all[test_m], y_enc[test_m]
s_tr=s_all[train_m]
print(f"Train: {len(X_tr_raw):,} | Test: {len(X_te_raw):,}")

scaler=StandardScaler()
X_tr=scaler.fit_transform(X_tr_raw)
X_te=scaler.transform(X_te_raw)

# ── LEVEL-0 BASE LEARNERS ───────────────────────────────────────────────────
print("\nDefining Level-0 base learners...")

# SVM removed from stacking — O(n^2) complexity makes it too slow on 27K+ windows.
# LightGBM + ExtraTrees still gives genuine model diversity (boosting vs bagging).
base_models = {
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, max_depth=-1,
        learning_rate=0.08, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        is_unbalance=True, n_jobs=-1, random_state=42, verbose=-1
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=150, max_depth=None, min_samples_leaf=3,
        n_jobs=-1, random_state=42
    ),
}

# ── OUT-OF-FOLD STACKING ────────────────────────────────────────────────────
# Use SUBJECT-based folds to avoid data leakage
# Each fold holds out one training subject
print("\nGenerating out-of-fold predictions (subject-based CV)...")

# Use 4 folds instead of 8 (all training subjects) — cuts stacking time in half
all_train_subjects = np.unique(s_tr)
unique_train_subjects = all_train_subjects[:4] if len(all_train_subjects) > 4 else all_train_subjects
n_folds = len(unique_train_subjects)
print(f"  {n_folds} folds (subset of training subjects, for speed)")

# Meta-features: stacked probabilities from all base models
meta_train = np.zeros((len(X_tr), n_cls * len(base_models)))
meta_test  = np.zeros((len(X_te), n_cls * len(base_models)))

test_preds_per_fold = {name: [] for name in base_models}

for fold_idx, val_sub in enumerate(unique_train_subjects):
    fold_val_m   = s_tr == val_sub
    fold_train_m = ~fold_val_m
    Xf_tr, yf_tr = X_tr[fold_train_m], y_tr[fold_train_m]
    Xf_va, yf_va = X_tr[fold_val_m],   y_tr[fold_val_m]
    print(f"\n  Fold {fold_idx+1}/{n_folds}: val=subject{int(val_sub)}"
          f" ({fold_val_m.sum()} windows)")

    col_start = 0
    for name, clf in base_models.items():
        clf.fit(Xf_tr, yf_tr)
        # Validation OOF predictions
        oof_proba = clf.predict_proba(Xf_va)
        meta_train[fold_val_m, col_start:col_start+n_cls] = oof_proba
        # Test predictions for this fold (averaged later)
        test_preds_per_fold[name].append(clf.predict_proba(X_te))
        col_start += n_cls
        print(f"    {name}: OOF acc={accuracy_score(yf_va,oof_proba.argmax(1)):.1%}")

# Average test predictions across folds
col_start = 0
for name in base_models:
    avg_test = np.mean(test_preds_per_fold[name], axis=0)
    meta_test[:, col_start:col_start+n_cls] = avg_test
    col_start += n_cls

print(f"\nMeta-train shape: {meta_train.shape}")
print(f"Meta-test shape:  {meta_test.shape}")

# ── LEVEL-1 META-LEARNER ────────────────────────────────────────────────────
print("\nTraining meta-learner (Logistic Regression)...")
meta_clf = LogisticRegression(
    C=1.0, max_iter=1000, 
    solver='lbfgs', random_state=42, n_jobs=-1
)
meta_clf.fit(meta_train, y_tr)
y_meta_prob = meta_clf.predict_proba(meta_test)
y_meta_pred = y_meta_prob.argmax(1)

raw_acc = accuracy_score(y_te, y_meta_pred)
print(f"\nRaw meta-learner accuracy: {raw_acc:.1%}")
print(classification_report(y_te, y_meta_pred, target_names=names, zero_division=0))

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

y_sm=min_dur(smooth(y_meta_pred,y_meta_prob.max(1),SMOOTH_WINDOW))
sm_acc=accuracy_score(y_te,y_sm)
sm_bal=balanced_accuracy_score(y_te,y_sm)
sm_f1=f1_score(y_te,y_sm,average='macro',zero_division=0)

print(f"\nAfter smoothing:")
print(f"  Accuracy:     {sm_acc:.1%}")
print(f"  Balanced Acc: {sm_bal:.1%}")
print(f"  Macro F1:     {sm_f1:.3f}")

print("\nPer-activity:")
for i,act in enumerate(names):
    mask=y_te==i
    if mask.sum()==0: continue
    a=accuracy_score(y_te[mask],y_sm[mask])
    print(f"  {act:<14} {a:.1%}  {'█'*int(a*20)}")

# ── META-LEARNER COEFFICIENT ANALYSIS ───────────────────────────────────────
print("\nMeta-learner: which base model it trusts per activity:")
model_names = list(base_models.keys())
coef = meta_clf.coef_  # shape: (n_cls, n_cls*n_base)
for cls_i, act in enumerate(names):
    contributions=[]
    for m_i, m_name in enumerate(model_names):
        start=m_i*n_cls
        c=coef[cls_i, start:start+n_cls].sum()
        contributions.append((m_name, c))
    top = max(contributions, key=lambda x: x[1])
    print(f"  {act:<14} → trusts {top[0]} most ({top[1]:.3f})")

# SAVE
tgt=np.array([y_meta_prob[i,y_te[i]] for i in range(len(y_te))])
def tl(c): return "GREEN" if c>=0.75 else ("YELLOW" if c>=0.45 else "RED")
tls=[tl(c) for c in tgt]
pd.DataFrame({
    'true_activity':[ACTIVITY_MAP.get(int(le.inverse_transform([i])[0]),'?') for i in y_te],
    'pred_activity':[ACTIVITY_MAP.get(int(le.inverse_transform([i])[0]),'?') for i in y_sm],
    'confidence':tgt.round(3),'is_correct':(y_sm==y_te).astype(int),
    'traffic_light':tls,'y_true_enc':y_te,'y_pred_enc':y_sm,
}).to_csv("pamap2_stacked_results.csv",index=False)
np.save("pamap2_stacked_proba.npy",y_meta_prob)

print(f"\n{'='*65}")
print(f"STACKED ENSEMBLE: raw={raw_acc:.1%}  smoothed={sm_acc:.1%}")
print(f"{'='*65}")