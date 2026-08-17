"""
PAMAP2 — HAR Transformer — FAST VERSION
Optimised for CPU-only laptops (i7, no GPU, 16GB RAM)

Speed fixes vs original:
  - 2 transformer blocks instead of 4 (attention is O(n²), this matters a lot)
  - d_model=32 instead of 64
  - 50 epochs instead of 100, patience 8 instead of 20
  - 2x augmentation instead of 3x

Expected time on i7-1185G7: ~20-30 min (vs 2-3 hours original)
Expected accuracy: 89-93%
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, LayerNormalization,
        GlobalAveragePooling1D, MultiHeadAttention, Add
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.losses import CategoricalCrossentropy
    print(f"TensorFlow {tf.__version__} loaded")
except ImportError:
    print("pip install tensorflow"); raise

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── CONFIG (SPEED-OPTIMISED) ──────────────────────────────────────────────
INPUT_FILE    = "pamap2_combined.csv"
WINDOW_SIZE   = 100
STEP_SIZE     = 50
TEST_SUBJECT  = 106
VAL_SUBJECT   = 108
EPOCHS        = 50          # was 100
BATCH_SIZE    = 256
SMOOTH_WINDOW = 25
PURITY_THR    = 0.85

D_MODEL  = 32   # was 64
N_HEADS  = 4
N_LAYERS = 2    # was 4
D_FF     = 64   # was 128
DROPOUT  = 0.3

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
print("PAMAP2 — HAR TRANSFORMER (FAST VERSION)")
print("="*65)

df = pd.read_csv(INPUT_FILE)
df = df[df['activityID'].isin(ACTIVITY_MAP.keys())]
SENSOR_COLS = [c for c in SENSOR_COLS if c in df.columns]
n_ch = len(SENSOR_COLS)
print(f"Shape: {df.shape} | Channels: {n_ch}")

def extract_windows(data):
    X,y,s=[],[],[]
    sd=data[SENSOR_COLS].values.astype(np.float32)
    ad=data['activityID'].values
    subj=int(data['subject_id'].iloc[0])
    disc=0
    for i in range(0,len(data)-WINDOW_SIZE,STEP_SIZE):
        lbs=ad[i:i+WINDOW_SIZE]; lab=pd.Series(lbs).mode()[0]
        if (lbs==lab).mean()<PURITY_THR: disc+=1; continue
        win=sd[i:i+WINDOW_SIZE]
        if np.isnan(win).any(): disc+=1; continue
        X.append(win); y.append(lab); s.append(subj)
    if disc: print(f"    Discarded {disc:,} impure")
    return np.array(X), np.array(y), np.array(s)

print("\nExtracting windows...")
all_X,all_y,all_s=[],[],[]
for sid in sorted(df['subject_id'].unique()):
    print(f"  Subject {int(sid)}...")
    sdf=df[df['subject_id']==sid].reset_index(drop=True)
    Xs,ys,ss=extract_windows(sdf)
    all_X.append(Xs); all_y.append(ys); all_s.append(ss)

X_all=np.concatenate(all_X); y_all=np.concatenate(all_y); s_all=np.concatenate(all_s)
print(f"Windows: {X_all.shape}")

le=LabelEncoder(); le.fit(sorted(ACTIVITY_MAP.keys()))
y_enc=le.transform(y_all); n_cls=len(le.classes_)
names=[ACTIVITY_MAP[int(c)] for c in le.classes_]

test_m=s_all==TEST_SUBJECT; val_m=s_all==VAL_SUBJECT; train_m=~test_m&~val_m
X_tr,y_tr=X_all[train_m],y_enc[train_m]
X_va,y_va=X_all[val_m],  y_enc[val_m]
X_te,y_te=X_all[test_m], y_enc[test_m]
print(f"Train: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,}")

ch_mean=X_tr.reshape(-1,n_ch).mean(0)
ch_std=np.where(X_tr.reshape(-1,n_ch).std(0)<1e-8,1.,X_tr.reshape(-1,n_ch).std(0))
X_tr=(X_tr-ch_mean)/ch_std; X_va=(X_va-ch_mean)/ch_std; X_te=(X_te-ch_mean)/ch_std

# 2x augmentation (was 3x)
X_aug = np.concatenate([X_tr, X_tr+np.random.normal(0,0.05,X_tr.shape).astype(np.float32)])
y_aug = np.concatenate([y_tr, y_tr])
idx = np.random.permutation(len(X_aug))
X_aug = X_aug[idx]; y_aug_c = to_categorical(y_aug[idx],n_cls)
y_va_c = to_categorical(y_va,n_cls)
print(f"Augmented train: {len(X_aug):,}")

def positional_encoding(length, d_model):
    positions = np.arange(length)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]
    angles    = positions / np.power(10000, (2*(dims//2))/d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)

def transformer_block(x, d_model, n_heads, d_ff, dropout_rate):
    attn_out = MultiHeadAttention(num_heads=n_heads, key_dim=d_model//n_heads,
                                   dropout=dropout_rate)(x, x)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn_out]))
    ff  = Dense(d_ff, activation='relu')(x)
    ff  = Dropout(dropout_rate)(ff)
    ff  = Dense(d_model)(ff)
    x   = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))
    return x

def build_transformer(T, C, n, d_model, n_heads, n_layers, d_ff, dropout):
    inp = Input(shape=(T, C))
    x = Dense(d_model)(inp)
    x = x + positional_encoding(T, d_model)
    x = Dropout(dropout)(x)
    for _ in range(n_layers):
        x = transformer_block(x, d_model, n_heads, d_ff, dropout)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    out = Dense(n, activation='softmax')(x)
    return Model(inp, out, name='HAR_Transformer_fast')

print("\nBuilding Transformer...")
model = build_transformer(WINDOW_SIZE, n_ch, n_cls, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT)
model.summary()
print(f"Total params: {model.count_params():,}")

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=EPOCHS*(len(X_aug)//BATCH_SIZE), alpha=1e-5)
model.compile(optimizer=Adam(lr_schedule),
              loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
    ModelCheckpoint('pamap2_transformer_best.weights.h5', monitor='val_accuracy',
                    save_best_only=True, save_weights_only=True, verbose=0),
]

print(f"\nTraining Transformer ({EPOCHS} epochs max, patience=8)...")
history = model.fit(X_aug, y_aug_c, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     validation_data=(X_va, y_va_c), callbacks=callbacks, verbose=1)
print(f"Best val accuracy: {max(history.history['val_accuracy']):.1%}")

print("\nEvaluating on Subject 106...")
y_prob = model.predict(X_te, batch_size=256, verbose=0)
y_pred = y_prob.argmax(1); conf = y_prob.max(1)
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

def min_dur(preds,mn=8):
    s=list(preds);n=len(s);changed=True;p=0
    while changed and p<5:
        changed=False;p+=1;i=0
        while i<n:
            curr=s[i];j=i+1
            while j<n and s[j]==curr: j+=1
            if j-i<mn:
                L=s[i-1] if i>0 else None; R=s[j] if j<n else None
                if L is not None and R is not None and L==R:
                    for k in range(i,j): s[k]=L
                    changed=True
            i=j
    return np.array(s)

y_sm   = min_dur(temporal_smooth(y_pred, conf, SMOOTH_WINDOW))
sm_acc = accuracy_score(y_te, y_sm)
sm_bal = balanced_accuracy_score(y_te, y_sm)
sm_f1  = f1_score(y_te, y_sm, average='macro', zero_division=0)
print(f"\nSmoothed: {sm_acc:.1%} | Balanced: {sm_bal:.1%} | F1: {sm_f1:.3f}")
print("\nPer-activity:")
for i,act in enumerate(names):
    mask=y_te==i
    if mask.sum()==0: continue
    a=accuracy_score(y_te[mask],y_sm[mask])
    print(f"  {act:<14} {a:.1%}  {'█'*int(a*20)}")

tgt_probs = np.array([y_prob[i,y_te[i]] for i in range(len(y_te))])
is_corr   = (y_sm==y_te).astype(int)
def tl(c): return "GREEN" if c>=0.75 else ("YELLOW" if c>=0.45 else "RED")
tls = [tl(c) for c in tgt_probs]
for label in ["GREEN","YELLOW","RED"]:
    n=tls.count(label); print(f"  {label}: {n:,} ({n/len(tls):.1%})")

model.save("pamap2_transformer.keras")
pd.DataFrame({
    'true_activity':[ACTIVITY_MAP.get(int(le.inverse_transform([i])[0]),'?') for i in y_te],
    'pred_activity':[ACTIVITY_MAP.get(int(le.inverse_transform([i])[0]),'?') for i in y_sm],
    'confidence':tgt_probs.round(3),
    'is_correct':is_corr,
    'traffic_light':tls,
    'y_true_enc':y_te,
    'y_pred_enc':y_sm,
}).to_csv("pamap2_transformer_results.csv", index=False)
np.save("pamap2_transformer_proba.npy", y_prob)

print(f"\n{'='*65}")
print(f"Transformer (FAST): raw={raw_acc:.1%}  smoothed={sm_acc:.1%}")
print(f"{'='*65}")
print("Saved: pamap2_transformer.keras  pamap2_transformer_results.csv")