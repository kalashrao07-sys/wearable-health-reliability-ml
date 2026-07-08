"""
PAMAP2 — CNN-BiLSTM Deep Learning Model
For: Wearable Sensor Reliability Detection

Architecture:
  Input: raw sensor windows (100 timesteps × 27 sensor channels)
  → Multi-scale CNN (local pattern extraction at 3 frequencies)
  → BiLSTM (temporal sequence modelling, forward + backward)
  → Dense classification head → Softmax
  → Confidence score → Traffic light reliability score

Why this beats classical ML:
  - Works on RAW data — no manual feature engineering
  - CNN learns which temporal patterns matter (ironing stroke rhythm vs static standing)
  - BiLSTM captures long-range temporal dependencies
  - Directly addresses the standing/ironing confusion

Run AFTER pamap2_preprocessing.py.
Install: pip install tensorflow
Output: pamap2_dl_model.h5 + pamap2_dl_reliability_results.csv
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
        Input, Conv1D, BatchNormalization, Activation, MaxPooling1D,
        Bidirectional, LSTM, Dropout, Dense, Concatenate
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    print(f"TensorFlow {tf.__version__} loaded")
except ImportError:
    print("Install TensorFlow:  pip install tensorflow")
    raise

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, classification_report)
from sklearn.preprocessing import LabelEncoder

# CONFIG
INPUT_FILE    = "pamap2_combined.csv"
WINDOW_SIZE   = 100
STEP_SIZE     = 50
TEST_SUBJECT  = 106
EPOCHS        = 80
BATCH_SIZE    = 256
SMOOTH_WINDOW = 25
PURITY_THR    = 0.85

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
print("PAMAP2 — CNN-BiLSTM DEEP LEARNING MODEL")
print("="*65)

# LOAD DATA
print("\nLoading data...")
df = pd.read_csv(INPUT_FILE)
df = df[df['activityID'].isin(ACTIVITY_MAP.keys())]
SENSOR_COLS = [c for c in SENSOR_COLS if c in df.columns]
print(f"  Shape: {df.shape} | Channels: {len(SENSOR_COLS)}")

# RAW WINDOW EXTRACTION
def extract_raw_windows(data):
    X_list, y_list, sub_list = [], [], []
    sensor_data   = data[SENSOR_COLS].values.astype(np.float32)
    activity_data = data['activityID'].values
    subj          = int(data['subject_id'].iloc[0])
    discarded     = 0
    for start in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        labels = activity_data[start:start+WINDOW_SIZE]
        label  = pd.Series(labels).mode()[0]
        if (labels==label).mean() < PURITY_THR:
            discarded += 1; continue
        win = sensor_data[start:start+WINDOW_SIZE]
        if np.isnan(win).any():
            discarded += 1; continue
        X_list.append(win); y_list.append(label); sub_list.append(subj)
    if discarded: print(f"    Discarded {discarded:,} impure windows")
    return np.array(X_list), np.array(y_list), np.array(sub_list)

print("\nExtracting raw windows...")
all_X, all_y, all_sub = [], [], []
for sid in sorted(df['subject_id'].unique()):
    print(f"  Subject {int(sid)}...")
    sdf = df[df['subject_id']==sid].reset_index(drop=True)
    Xs,ys,ss = extract_raw_windows(sdf)
    all_X.append(Xs); all_y.append(ys); all_sub.append(ss)

X_all   = np.concatenate(all_X)
y_all   = np.concatenate(all_y)
sub_all = np.concatenate(all_sub)
print(f"\n  Windows: {X_all.shape[0]:,} × {WINDOW_SIZE} timesteps × {len(SENSOR_COLS)} channels")

# LABEL ENCODING
le = LabelEncoder()
le.fit(sorted(ACTIVITY_MAP.keys()))
y_enc   = le.transform(y_all)
n_cls   = len(le.classes_)
act_names = [ACTIVITY_MAP[int(c)] for c in le.classes_]
print(f"  Classes ({n_cls}): {act_names}")

# TRAIN / TEST SPLIT
test_m  = (sub_all == TEST_SUBJECT)
train_m = ~test_m
X_tr_r, y_tr_r = X_all[train_m], y_enc[train_m]
X_te_r, y_te_r = X_all[test_m],  y_enc[test_m]
print(f"\nTrain: {len(X_tr_r):,} | Test: {len(X_te_r):,} (subject {TEST_SUBJECT})")

# CHANNEL NORMALISATION
n_ch     = len(SENSOR_COLS)
ch_mean  = X_tr_r.reshape(-1,n_ch).mean(0)
ch_std   = X_tr_r.reshape(-1,n_ch).std(0)
ch_std   = np.where(ch_std<1e-8, 1.0, ch_std)
X_tr     = (X_tr_r - ch_mean) / ch_std
X_te     = (X_te_r - ch_mean) / ch_std
np.save("pamap2_dl_channel_mean.npy", ch_mean)
np.save("pamap2_dl_channel_std.npy",  ch_std)

y_tr_cat = to_categorical(y_tr_r, n_cls)
y_te_cat = to_categorical(y_te_r, n_cls)

# CNN-BiLSTM ARCHITECTURE
def build_model(T, C, n):
    inp = Input(shape=(T,C), name='input')

    # Multi-scale CNN branches
    k3 = Conv1D(64,3,padding='same')(inp)
    k3 = BatchNormalization()(k3); k3 = Activation('relu')(k3)
    k3 = Conv1D(64,3,padding='same')(k3)
    k3 = BatchNormalization()(k3); k3 = Activation('relu')(k3)

    k7 = Conv1D(64,7,padding='same')(inp)
    k7 = BatchNormalization()(k7); k7 = Activation('relu')(k7)
    k7 = Conv1D(64,7,padding='same')(k7)
    k7 = BatchNormalization()(k7); k7 = Activation('relu')(k7)

    x = Concatenate()([k3,k7])

    x = Conv1D(128,3,padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling1D(2)(x); x = Dropout(0.2)(x)

    x = Conv1D(256,3,padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling1D(2)(x); x = Dropout(0.2)(x)

    # BiLSTM
    x = Bidirectional(LSTM(128,return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64,return_sequences=False))(x)
    x = Dropout(0.3)(x)

    # Head
    x = Dense(128,activation='relu')(x)
    x = BatchNormalization()(x); x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(n, activation='softmax', name='output')(x)
    return Model(inp, out, name='CNN_BiLSTM')

print("\nBuilding model...")
model = build_model(WINDOW_SIZE, n_ch, n_cls)
model.summary()
print(f"  Total params: {model.count_params():,}")

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
    ModelCheckpoint('pamap2_dl_best.weights.h5', monitor='val_accuracy',
                    save_best_only=True, save_weights_only=True, verbose=0),
]

print(f"\nTraining ({EPOCHS} epochs max, early stop patience=15)...")
history = model.fit(
    X_tr, y_tr_cat,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_split=0.15, callbacks=callbacks, verbose=1
)

# EVALUATION
print("\nEvaluating on Subject 106 test set...")
y_prob = model.predict(X_te, batch_size=256, verbose=0)
y_pred = y_prob.argmax(1)
conf   = y_prob.max(1)

raw_acc = accuracy_score(y_te_r, y_pred)
raw_f1  = f1_score(y_te_r, y_pred, average='macro', zero_division=0)
print(f"\n  Raw accuracy: {raw_acc:.1%} | Macro F1: {raw_f1:.3f}")
print(classification_report(y_te_r,y_pred,target_names=act_names,zero_division=0))

# TEMPORAL SMOOTHING (same as classical ML)
def temporal_smooth(preds, confs, w=25):
    out = preds.copy(); half = w//2
    for i in range(len(preds)):
        lo,hi = max(0,i-half), min(len(preds),i+half+1)
        counts = {}
        for j in range(lo,hi):
            v=preds[j]; counts[v]=counts.get(v,0)+float(confs[j])
        out[i] = max(counts,key=counts.get)
    return out

def min_duration(preds, mn=8):
    s=list(preds); n=len(s); changed=True; p=0
    while changed and p<5:
        changed=False; p+=1; i=0
        while i<n:
            curr=s[i]; j=i+1
            while j<n and s[j]==curr: j+=1
            if j-i<mn:
                L=s[i-1] if i>0 else None
                R=s[j]   if j<n else None
                if L is not None and R is not None and L==R:
                    for k in range(i,j): s[k]=L
                    changed=True
            i=j
    return np.array(s)

y_sm   = temporal_smooth(y_pred, conf, SMOOTH_WINDOW)
y_sm   = min_duration(y_sm)
sm_acc = accuracy_score(y_te_r, y_sm)
sm_f1  = f1_score(y_te_r, y_sm, average='macro', zero_division=0)
sm_bal = balanced_accuracy_score(y_te_r, y_sm)

print(f"\nAfter temporal smoothing + min duration:")
print(f"  Accuracy:     {sm_acc:.1%}")
print(f"  Balanced Acc: {sm_bal:.1%}")
print(f"  Macro F1:     {sm_f1:.3f}")

print("\nPer-activity (smoothed):")
for i,act in enumerate(act_names):
    mask = y_te_r==i
    if mask.sum()==0: continue
    a = accuracy_score(y_te_r[mask],y_sm[mask])
    bar = '█'*int(a*20)
    print(f"  {act:<14} {a:.1%}  {bar}")

# RELIABILITY SCORING
target_probs = np.array([y_prob[i,y_te_r[i]] for i in range(len(y_te_r))])
is_correct   = (y_sm==y_te_r).astype(int)

def tl(c):
    if c>=0.75: return "GREEN"
    if c>=0.45: return "YELLOW"
    return "RED"

tls    = [tl(c) for c in target_probs]
total  = len(tls)
green  = tls.count("GREEN")
yellow = tls.count("YELLOW")
red    = tls.count("RED")

print(f"\nReliability (DL model):")
print(f"  GREEN  : {green:,} ({green/total:.1%})")
print(f"  YELLOW : {yellow:,} ({yellow/total:.1%})")
print(f"  RED    : {red:,} ({red/total:.1%})")

# LOSO
print(f"\nLeave-One-Subject-Out Cross-Validation...")
loso = {}
for test_sub in sorted(np.unique(sub_all)):
    tm  = sub_all==test_sub
    trm = ~tm
    Xtr_l = X_all[trm]; ytr_l = y_enc[trm]
    Xte_l = X_all[tm];  yte_l = y_enc[tm]
    if len(Xte_l)==0: print(f"  Sub {int(test_sub)}: Skipped"); continue
    cm = Xtr_l.reshape(-1,n_ch).mean(0)
    cs = Xtr_l.reshape(-1,n_ch).std(0)
    cs = np.where(cs<1e-8,1.0,cs)
    Xtr_n=(Xtr_l-cm)/cs; Xte_n=(Xte_l-cm)/cs
    lm = build_model(WINDOW_SIZE,n_ch,n_cls)
    lm.compile(optimizer=Adam(1e-3),loss='categorical_crossentropy',metrics=['accuracy'])
    lm.fit(Xtr_n,to_categorical(ytr_l,n_cls),epochs=50,batch_size=BATCH_SIZE,
           validation_split=0.1,
           callbacks=[EarlyStopping(monitor='val_accuracy',patience=10,restore_best_weights=True,verbose=0)],
           verbose=0)
    yp    = lm.predict(Xte_n,verbose=0)
    ypsm  = temporal_smooth(yp.argmax(1),yp.max(1),SMOOTH_WINDOW)
    acc   = accuracy_score(yte_l,ypsm)
    bal   = balanced_accuracy_score(yte_l,ypsm)
    loso[int(test_sub)] = {"accuracy":round(acc,4),"balanced":round(bal,4)}
    print(f"  Subject {int(test_sub)}: {acc:.1%} (balanced: {bal:.1%})")
    tf.keras.backend.clear_session()

lavg = np.mean([v['accuracy'] for v in loso.values()])
lstd = np.std( [v['accuracy'] for v in loso.values()])
lbal = np.mean([v['balanced'] for v in loso.values()])
print(f"  Average: {lavg:.1%} ± {lstd:.1%} | Balanced: {lbal:.1%}")

# SAVE
model.save("pamap2_dl_model.h5")
pd.DataFrame({
    'true_activity':[ACTIVITY_MAP.get(int(le.inverse_transform([i])[0]),'?') for i in y_te_r],
    'pred_activity':[ACTIVITY_MAP.get(int(le.inverse_transform([i])[0]),'?') for i in y_sm],
    'confidence':target_probs.round(3),
    'is_correct':is_correct,
    'traffic_light':tls,
}).to_csv("pamap2_dl_reliability_results.csv",index=False)

with open("pamap2_dl_label_map.json","w") as f:
    json.dump({"classes":le.classes_.tolist(),
               "activity_map":{str(k):v for k,v in ACTIVITY_MAP.items()}},f)

pd.DataFrame([{"subject":k,"accuracy":v["accuracy"],"balanced":v["balanced"]}
              for k,v in loso.items()]).to_csv("pamap2_dl_loso.csv",index=False)

print(f"\n{'='*65}")
print("COMPARISON SUMMARY")
print(f"{'='*65}")
print(f"{'Model':<35} {'Raw Acc':>10} {'Smoothed':>10} {'LOSO':>10}")
print(f"{'-'*65}")
print(f"{'Voting Ensemble (RF+KNN+XGB)':<35} {'92.3%':>10} {'95.8%':>10} {'91.5%':>10}")
print(f"{'CNN-BiLSTM':<35} {raw_acc:>9.1%} {sm_acc:>9.1%} {lavg:>9.1%}")
print(f"{'='*65}")
print(f"\nSaved: pamap2_dl_model.h5")
print(f"       pamap2_dl_reliability_results.csv")
print(f"       pamap2_dl_loso.csv")
print(f"       pamap2_dl_label_map.json")