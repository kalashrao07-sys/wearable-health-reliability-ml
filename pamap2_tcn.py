"""
PAMAP2 — Temporal Convolutional Network (TCN)
For: Wearable Sensor Reliability Detection

Why TCN over CNN-BiLSTM:
  - Dilated causal convolutions grow receptive field EXPONENTIALLY
    with depth — dilation [1,2,4,8,16] covers 32 timesteps per layer,
    stacked to cover full 100-step window with fewer parameters
  - No sequential bottleneck (unlike LSTM) — fully parallelisable
  - Residual connections allow direct gradient flow across all layers
  - Captures multi-scale frequency patterns simultaneously:
      dilation 1  → high-frequency micro-movements (heel strike impact)
      dilation 4  → gait cycle components (~0.25s)
      dilation 16 → full stride or ironing stroke (~1.0s)
  - Has outperformed LSTM on 8/9 standard time-series benchmarks

Architecture:
  Input (100, 27)
  → 4 × TCN Residual Block (Conv1D dilated + skip + BN + ReLU)
  → GlobalAveragePooling1D
  → Dense head → Softmax (11 classes)

Install: pip install tensorflow
Run AFTER pamap2_preprocessing.py
"""

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv1D, BatchNormalization, Activation,
        Add, Dropout, Dense, GlobalAveragePooling1D,
        SpatialDropout1D
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.losses import CategoricalCrossentropy
    print(f"TensorFlow {tf.__version__} loaded")
except ImportError:
    print("Install: pip install tensorflow"); raise

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, classification_report)
from sklearn.preprocessing import LabelEncoder

# ── CONFIG ─────────────────────────────────────────────────────────────────
INPUT_FILE    = "pamap2_combined.csv"
WINDOW_SIZE   = 100
STEP_SIZE     = 50
TEST_SUBJECT  = 106
VAL_SUBJECT   = 108
EPOCHS        = 120
BATCH_SIZE    = 256
SMOOTH_WINDOW = 25
PURITY_THR    = 0.85

# TCN hyperparameters
N_FILTERS      = 64        # convolutional filters per block
KERNEL_SIZE    = 3         # temporal kernel size
DILATIONS      = [1,2,4,8,16]  # exponentially growing receptive field
DROPOUT_RATE   = 0.25      # SpatialDropout1D (drops entire feature maps)

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
print("PAMAP2 — TEMPORAL CONVOLUTIONAL NETWORK (TCN)")
print("="*65)

# ── LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
df = df[df['activityID'].isin(ACTIVITY_MAP.keys())]
SENSOR_COLS = [c for c in SENSOR_COLS if c in df.columns]
n_ch = len(SENSOR_COLS)
print(f"Shape: {df.shape} | Channels: {n_ch}")

# ── WINDOW EXTRACTION ───────────────────────────────────────────────────────
def extract_windows(data):
    X,y,s=[],[],[]
    sd = data[SENSOR_COLS].values.astype(np.float32)
    ad = data['activityID'].values
    subj = int(data['subject_id'].iloc[0])
    disc = 0
    for i in range(0, len(data)-WINDOW_SIZE, STEP_SIZE):
        lbs = ad[i:i+WINDOW_SIZE]
        lab = pd.Series(lbs).mode()[0]
        if (lbs==lab).mean() < PURITY_THR: disc+=1; continue
        win = sd[i:i+WINDOW_SIZE]
        if np.isnan(win).any(): disc+=1; continue
        X.append(win); y.append(lab); s.append(subj)
    if disc: print(f"    Discarded {disc:,} impure windows")
    return np.array(X), np.array(y), np.array(s)

print("\nExtracting windows...")
all_X,all_y,all_s=[],[],[]
for sid in sorted(df['subject_id'].unique()):
    print(f"  Subject {int(sid)}...")
    sdf = df[df['subject_id']==sid].reset_index(drop=True)
    Xs,ys,ss = extract_windows(sdf)
    all_X.append(Xs); all_y.append(ys); all_s.append(ss)

X_all=np.concatenate(all_X); y_all=np.concatenate(all_y); s_all=np.concatenate(all_s)
print(f"Windows: {X_all.shape}")

le = LabelEncoder()
le.fit(sorted(ACTIVITY_MAP.keys()))
y_enc  = le.transform(y_all)
n_cls  = len(le.classes_)
names  = [ACTIVITY_MAP[int(c)] for c in le.classes_]

# Subject-based splits
test_m = s_all==TEST_SUBJECT; val_m = s_all==VAL_SUBJECT; train_m=~test_m&~val_m
X_tr,y_tr = X_all[train_m],y_enc[train_m]
X_va,y_va = X_all[val_m],  y_enc[val_m]
X_te,y_te = X_all[test_m], y_enc[test_m]
print(f"Train: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,}")
print(f"Train subjects: {sorted(set(s_all[train_m].tolist()))}")

# Normalise
n_ch_     = n_ch
ch_mean   = X_tr.reshape(-1,n_ch_).mean(0)
ch_std    = np.where(X_tr.reshape(-1,n_ch_).std(0)<1e-8, 1., X_tr.reshape(-1,n_ch_).std(0))
X_tr=(X_tr-ch_mean)/ch_std; X_va=(X_va-ch_mean)/ch_std; X_te=(X_te-ch_mean)/ch_std
np.save("pamap2_tcn_ch_mean.npy", ch_mean)
np.save("pamap2_tcn_ch_std.npy",  ch_std)

# Augmentation (3×)
X_aug = np.concatenate([
    X_tr,
    X_tr + np.random.normal(0, 0.05, X_tr.shape).astype(np.float32),
    X_tr * np.random.uniform(0.85,1.15,(len(X_tr),1,n_ch_)).astype(np.float32),
])
y_aug   = np.concatenate([y_tr]*3)
idx     = np.random.permutation(len(X_aug))
X_aug   = X_aug[idx]
y_aug_c = to_categorical(y_aug[idx], n_cls)
y_va_c  = to_categorical(y_va, n_cls)
y_te_c  = to_categorical(y_te, n_cls)
print(f"Augmented train: {len(X_aug):,}")

# ── TCN RESIDUAL BLOCK ──────────────────────────────────────────────────────
def tcn_residual_block(x, n_filters, kernel_size, dilation, dropout):
    """
    One TCN residual block:
      - Two dilated causal Conv1D layers with BN + ReLU
      - SpatialDropout1D (drops entire feature channels, not random timesteps)
      - 1×1 Conv skip connection to match dimensions
      - Add residual

    Causal padding ensures no future information leaks:
      output[t] depends only on input[0..t]
    """
    padding = (kernel_size - 1) * dilation

    # Branch 1: dilated causal convolutions
    x_out = tf.keras.layers.ZeroPadding1D((padding,0))(x)
    x_out = Conv1D(n_filters, kernel_size, dilation_rate=dilation,
                   padding='valid', activation=None)(x_out)
    x_out = BatchNormalization()(x_out)
    x_out = Activation('relu')(x_out)
    x_out = SpatialDropout1D(dropout)(x_out)

    x_out = tf.keras.layers.ZeroPadding1D((padding,0))(x_out)
    x_out = Conv1D(n_filters, kernel_size, dilation_rate=dilation,
                   padding='valid', activation=None)(x_out)
    x_out = BatchNormalization()(x_out)
    x_out = Activation('relu')(x_out)
    x_out = SpatialDropout1D(dropout)(x_out)

    # Branch 2: 1×1 skip connection (match filter count)
    skip = Conv1D(n_filters, 1, padding='same')(x)

    return Activation('relu')(Add()([x_out, skip]))

# ── BUILD TCN ───────────────────────────────────────────────────────────────
def build_tcn(T, C, n):
    """
    Full TCN:
      4 residual blocks with dilations [1,2,4,8,16]
      Receptive field = sum((k-1)*d for d in dilations) * n_stacks
                     = (3-1)*(1+2+4+8+16) = 62 timesteps per stack
      Two stacks of 5 dilations = 124 timestep receptive field > 100 needed
    """
    inp = Input(shape=(T,C), name='input')

    # Initial projection to n_filters
    x = Conv1D(N_FILTERS, 1, padding='same')(inp)

    # Stack of TCN residual blocks
    for d in DILATIONS:
        x = tcn_residual_block(x, N_FILTERS, KERNEL_SIZE, d, DROPOUT_RATE)
    # Second stack (deeper representation)
    for d in DILATIONS[:3]:  # [1,2,4] — partial second stack
        x = tcn_residual_block(x, N_FILTERS, KERNEL_SIZE, d, DROPOUT_RATE)

    # Global pooling — aggregate all timesteps
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(n, activation='softmax', name='output')(x)

    return Model(inp, out, name='TCN')

print("\nBuilding TCN...")
model = build_tcn(WINDOW_SIZE, n_ch_, n_cls)
model.summary()
print(f"Total params: {model.count_params():,}")

# Compute total receptive field
rf = sum((KERNEL_SIZE-1)*d for d in DILATIONS)*2 + sum((KERNEL_SIZE-1)*d for d in DILATIONS[:3])
print(f"Receptive field: ~{rf} timesteps = {rf/100:.2f}s")

# Cosine decay LR
lr_sched = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=EPOCHS*(len(X_aug)//BATCH_SIZE),
    alpha=1e-5
)
model.compile(
    optimizer=Adam(lr_sched),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint('pamap2_tcn_best.weights.h5', monitor='val_accuracy',
                    save_best_only=True, save_weights_only=True, verbose=0),
]

print(f"\nTraining TCN on {len(X_aug):,} augmented windows...")
history = model.fit(
    X_aug, y_aug_c, epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(X_va, y_va_c), callbacks=callbacks, verbose=1
)
print(f"Best val accuracy: {max(history.history['val_accuracy']):.1%}")

# ── EVALUATE ────────────────────────────────────────────────────────────────
print(f"\nEvaluating on Subject {TEST_SUBJECT}...")
y_prob = model.predict(X_te, batch_size=256, verbose=0)
y_pred = y_prob.argmax(1)
conf   = y_prob.max(1)
raw_acc = accuracy_score(y_te, y_pred)
print(f"Raw: {raw_acc:.1%}")
print(classification_report(y_te, y_pred, target_names=names, zero_division=0))

def temporal_smooth(preds, confs, w=25):
    out=preds.copy(); half=w//2
    for i in range(len(preds)):
        lo,hi=max(0,i-half),min(len(preds),i+half+1)
        counts={}
        for j in range(lo,hi): v=preds[j]; counts[v]=counts.get(v,0)+float(confs[j])
        out[i]=max(counts,key=counts.get)
    return out

def min_dur(preds, mn=8):
    s=list(preds);n=len(s);changed=True;p=0
    while changed and p<5:
        changed=False;p+=1;i=0
        while i<n:
            curr=s[i];j=i+1
            while j<n and s[j]==curr: j+=1
            if j-i<mn:
                L=s[i-1] if i>0 else None; R=s[j] if j<n else None
                if L and R and L==R:
                    for k in range(i,j): s[k]=L; changed=True
            i=j
    return np.array(s)

y_sm   = min_dur(temporal_smooth(y_pred, conf, SMOOTH_WINDOW))
sm_acc = accuracy_score(y_te, y_sm)
sm_bal = balanced_accuracy_score(y_te, y_sm)
sm_f1  = f1_score(y_te, y_sm, average='macro', zero_division=0)

print(f"\nAfter smoothing:")
print(f"  Accuracy:     {sm_acc:.1%}")
print(f"  Balanced Acc: {sm_bal:.1%}")
print(f"  Macro F1:     {sm_f1:.3f}")
print(f"\nPer-activity:")
for i,act in enumerate(names):
    mask=y_te==i
    if mask.sum()==0: continue
    a=accuracy_score(y_te[mask],y_sm[mask])
    print(f"  {act:<14} {a:.1%}  {'█'*int(a*20)}")

# Reliability
tgt_probs = np.array([y_prob[i,y_te[i]] for i in range(len(y_te))])
is_corr   = (y_sm==y_te).astype(int)
def tl(c): return "GREEN" if c>=0.75 else ("YELLOW" if c>=0.45 else "RED")
tls=[tl(c) for c in tgt_probs]
for label in ["GREEN","YELLOW","RED"]:
    n_=tls.count(label); print(f"  {label}: {n_:,} ({n_/len(tls):.1%})")

# LOSO
print("\nLOSO Cross-Validation...")
loso={}
for test_sub in sorted(np.unique(s_all)):
    tm=s_all==test_sub; trm=~tm
    Xtr_l=X_all[trm]; ytr_l=y_enc[trm]
    Xte_l=X_all[tm];  yte_l=y_enc[tm]
    if len(Xte_l)==0: print(f"  Sub {int(test_sub)}: Skipped"); continue
    cm=Xtr_l.reshape(-1,n_ch_).mean(0)
    cs=np.where(Xtr_l.reshape(-1,n_ch_).std(0)<1e-8,1.,Xtr_l.reshape(-1,n_ch_).std(0))
    Xtr_n=(Xtr_l-cm)/cs; Xte_n=(Xte_l-cm)/cs
    lm=build_tcn(WINDOW_SIZE,n_ch_,n_cls)
    lm.compile(optimizer=Adam(1e-3),loss=CategoricalCrossentropy(label_smoothing=0.1),metrics=['accuracy'])
    lm.fit(Xtr_n,to_categorical(ytr_l,n_cls),epochs=60,batch_size=BATCH_SIZE,
           validation_split=0.1,
           callbacks=[EarlyStopping(monitor='val_accuracy',patience=12,restore_best_weights=True,verbose=0)],
           verbose=0)
    yp=lm.predict(Xte_n,verbose=0)
    ypsm=min_dur(temporal_smooth(yp.argmax(1),yp.max(1),SMOOTH_WINDOW))
    acc=accuracy_score(yte_l,ypsm); bal=balanced_accuracy_score(yte_l,ypsm)
    loso[int(test_sub)]={"accuracy":round(acc,4),"balanced":round(bal,4)}
    print(f"  Subject {int(test_sub)}: {acc:.1%}  (balanced: {bal:.1%})")
    tf.keras.backend.clear_session()

lavg=np.mean([v['accuracy'] for v in loso.values()])
lstd=np.std([v['accuracy'] for v in loso.values()])
lbal=np.mean([v['balanced'] for v in loso.values()])
print(f"  Average: {lavg:.1%} ± {lstd:.1%} | Balanced: {lbal:.1%}")

# SAVE
model.save("pamap2_tcn.keras")
pd.DataFrame({
    'true_activity':[ACTIVITY_MAP.get(int(le.inverse_transform([i])[0]),'?') for i in y_te],
    'pred_activity':[ACTIVITY_MAP.get(int(le.inverse_transform([i])[0]),'?') for i in y_sm],
    'confidence':tgt_probs.round(3),
    'is_correct':is_corr,
    'traffic_light':tls,
    'y_true_enc':y_te,
    'y_pred_enc':y_sm,
}).to_csv("pamap2_tcn_results.csv",index=False)
np.save("pamap2_tcn_proba.npy", y_prob)
pd.DataFrame([{"subject":k,"accuracy":v["accuracy"],"balanced":v["balanced"]}
              for k,v in loso.items()]).to_csv("pamap2_tcn_loso.csv",index=False)

print(f"\n{'='*65}")
print(f"TCN RESULT: raw={raw_acc:.1%}  smoothed={sm_acc:.1%}  LOSO={lavg:.1%}±{lstd:.1%}")
print(f"{'='*65}")
print(f"Saved: pamap2_tcn.keras  pamap2_tcn_results.csv  pamap2_tcn_proba.npy")
