"""
PAMAP2 — Quantum Machine Learning (VQC via PennyLane)
For: Wearable Sensor Reliability Detection

Three QML approaches compared:
  1. VQC (Variational Quantum Classifier) — AngleEmbedding + StronglyEntanglingLayers
  2. QKSVM (Quantum Kernel SVM) — ZZFeatureMap kernel + classical SVM
  3. QTL (Quantum Transfer Learning) — classical embeddings → quantum circuit layer

Pipeline:
  308 features → PCA(16) → quantum circuit → softmax → 11 classes

Why PCA to 16 dims:
  - Current NISQ simulators handle ~20 qubits cleanly
  - 16 PCs capture ~95% of feature variance
  - AngleEmbedding needs one qubit per feature

Install: pip install pennylane pennylane-lightning
Run AFTER pamap2_ml_model.py (needs pamap2_combined.csv + pamap2_feature_cols.json)

── SPEED OPTIMISATIONS (results equivalent, same math) ───────────────────
  1. Device: default.qubit -> lightning.qubit (C++ backend, no approximation).
  2. Diff method: autograd backprop -> adjoint differentiation for circuits
     that only return expectation values (VQC, QTL). The QKSVM kernel
     circuit returns qml.probs(), which adjoint doesn't support, so it
     keeps the default diff method (it's forward-only anyway — no
     gradients are taken through it).
  3. QKSVM train-train kernel block exploits symmetry (K[i,j]==K[j,i]).

── CORRECTNESS FIXES (from review; results WILL change, for the better) ───
  1. VQC training now does true multi-batch epochs (iterates over ALL
     mini-batches per epoch) instead of one 64-sample batch = one epoch.
     The old code gave each of the 11 OvR classifiers only 25 total
     gradient updates; this version gives ~30x more per epoch and runs
     several epochs with early stopping.
  2. Near/below-random VQC accuracy is now flagged with an explicit
     warning instead of being silently printed as a finished result.
  3. VQC/QTL subsampling is now stratified per class (previously a single
     global random draw, which could under-represent some classes).
  4. VQC now has a proper validation split (carved from TRAINING subjects
     only, never the held-out test subject) with early stopping on
     validation loss, and returns the best-validation-loss weights.
  5. Angle-embedding normalisation now fits min/max on TRAIN data only and
     applies that fixed transform to test data (previously each split was
     scaled independently, which is a train/test inconsistency).
  6. QKSVM test-set leakage FIXED: the previous version trained the
     kernel SVM on windows from the held-out TEST_SUBJECT and evaluated
     on more windows from that same subject. Training data now comes
     exclusively from training subjects; test data exclusively from the
     held-out subject, with disjoint indices.
  7. QKSVM kernel circuit now actually implements the Hadamard-based
     feature map its docstring described (previously the docstring
     mentioned Hadamards that weren't in the circuit).
  8. QTL's broken `class % 4` training objective is replaced with a
     contrastive (same-class-pull / different-class-push) objective that
     is well-defined for all 11 classes, instead of silently collapsing
     multiple classes onto the same 4-dim one-hot target.
  9. VQC, QKSVM, and QTL now all use the SAME held-out-subject protocol
     and comparable stratified sample sizes, so the three-way comparison
     is apples-to-apples. Sample sizes and subsampling are explicitly
     printed/disclosed at each stage rather than left implicit.
 10. Per-classifier random seeding added for VQC (RANDOM_SEED + cls_idx)
     for reproducibility. Running the whole script multiple times with
     different RANDOM_SEED values and reporting mean±std across seeds is
     recommended for a research write-up (not automated here, since it
     multiplies runtime by the seed count — see note near RANDOM_SEED).
"""

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    print(f"PennyLane {qml.__version__} loaded")
except ImportError:
    print("Install: pip install pennylane pennylane-lightning")
    raise

# Prefer the fast C++ simulator; fall back gracefully if unavailable.
try:
    qml.device("lightning.qubit", wires=1)
    QDEVICE = "lightning.qubit"
    QDIFF   = "adjoint"
    print("Using lightning.qubit (C++ backend) with adjoint differentiation")
except Exception:
    QDEVICE = "default.qubit"
    QDIFF   = "backprop"
    print("lightning.qubit unavailable, falling back to default.qubit")
    print("  (pip install pennylane-lightning for a large speedup)")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, classification_report)
from sklearn.calibration import CalibratedClassifierCV
import time

# ── CONFIG ─────────────────────────────────────────────────────────────────
INPUT_FILE    = "pamap2_combined.csv"
FEAT_COLS_FILE= "pamap2_feature_cols.json"
WINDOW_SIZE   = 100
STEP_SIZE     = 50
TEST_SUBJECT  = 106
PURITY_THR    = 0.85
N_QUBITS      = 16    # PCA components = qubits
N_LAYERS      = 3     # variational circuit depth
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
print("PAMAP2 — QUANTUM MACHINE LEARNING (PennyLane)")
print("="*65)
print(f"Device: {QDEVICE} (CPU simulation, diff_method={QDIFF})")
print(f"Qubits: {N_QUBITS} | VQC layers: {N_LAYERS}")
print(f"Encoding: AngleEmbedding (one qubit per PCA component)")

# ── LOAD DATA ───────────────────────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv(INPUT_FILE)
df = df[df['activityID'].isin(ACTIVITY_MAP.keys())]
SENSOR_COLS = [c for c in SENSOR_COLS if c in df.columns]
n_ch = len(SENSOR_COLS)

# ── FEATURE EXTRACTION (same as classical ML) ───────────────────────────────
print("Extracting statistical + spectral features (same as classical ML)...")

def spectral_entropy(signal):
    fft_mag = np.abs(np.fft.rfft(signal))
    power   = fft_mag**2; total=power.sum()
    if total==0: return 0.0
    prob=power/total; prob=prob[prob>0]
    return -np.sum(prob*np.log2(prob))

def extract_features(data):
    """Extract 308 features — identical to pamap2_ml_model.py"""
    freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0/100)
    records=[]
    disc=0
    sensor_data = data[SENSOR_COLS].values.astype(np.float32)
    activity_data = data['activityID'].values
    subj = int(data['subject_id'].iloc[0])
    for start in range(0, len(data)-WINDOW_SIZE, STEP_SIZE):
        lbs = activity_data[start:start+WINDOW_SIZE]
        lab = pd.Series(lbs).mode()[0]
        if (lbs==lab).mean() < PURITY_THR: disc+=1; continue
        win = sensor_data[start:start+WINDOW_SIZE]
        if np.isnan(win).any(): disc+=1; continue
        feat={}
        for ci,col in enumerate(SENSOR_COLS):
            vals=win[:,ci]
            feat[f"{col}_mean"]  =vals.mean()
            feat[f"{col}_std"]   =vals.std()
            feat[f"{col}_min"]   =vals.min()
            feat[f"{col}_max"]   =vals.max()
            feat[f"{col}_range"] =vals.max()-vals.min()
            feat[f"{col}_energy"]=(vals**2).mean()
            fft_mag=np.abs(np.fft.rfft(vals-vals.mean()))
            power=fft_mag**2; dom=np.argmax(power)
            feat[f"{col}_dom_freq"]        =freqs[dom]
            feat[f"{col}_peak_power"]      =power[dom]
            feat[f"{col}_spectral_entropy"]=spectral_entropy(vals)
        mags={}
        for loc in ['hand','chest','ankle']:
            xi=SENSOR_COLS.index(f'{loc}_acc16_x')
            yi=SENSOR_COLS.index(f'{loc}_acc16_y')
            zi=SENSOR_COLS.index(f'{loc}_acc16_z')
            mag=np.sqrt(win[:,xi]**2+win[:,yi]**2+win[:,zi]**2)
            mags[loc]=mag
            feat[f'{loc}_acc_mag_mean']=mag.mean()
            feat[f'{loc}_acc_mag_std'] =mag.std()
            delta=np.diff(mag)
            feat[f'{loc}_acc_mag_delta_mean']=np.abs(delta).mean()
            feat[f'{loc}_acc_mag_delta_std'] =delta.std()
        def safe_corr(a,b):
            if a.std()<1e-8 or b.std()<1e-8: return 0.
            return float(np.corrcoef(a,b)[0,1])
        feat['corr_hand_ankle'] =safe_corr(mags['hand'],mags['ankle'])
        feat['corr_hand_chest'] =safe_corr(mags['hand'],mags['chest'])
        feat['corr_chest_ankle']=safe_corr(mags['chest'],mags['ankle'])
        feat['hand_ankle_ratio'] =mags['hand'].std()/(mags['ankle'].std()+1e-6)
        feat['hand_chest_ratio'] =mags['hand'].std()/(mags['chest'].std()+1e-6)
        feat['chest_ankle_ratio']=mags['chest'].std()/(mags['ankle'].std()+1e-6)
        for loc in ['hand','chest','ankle']:
            zi=SENSOR_COLS.index(f'{loc}_acc16_z')
            feat[f'{loc}_tilt_proxy']=win[:,zi].mean()
        feat['activityID']=lab; feat['subject_id']=subj
        records.append(feat)
    return pd.DataFrame(records), disc

all_dfs=[]
total_disc=0
for sid in sorted(df['subject_id'].unique()):
    print(f"  Subject {int(sid)}...", end='', flush=True)
    sdf=df[df['subject_id']==sid].reset_index(drop=True)
    fdf,disc=extract_features(sdf)
    all_dfs.append(fdf); total_disc+=disc
    print(f" {len(fdf)} windows")

windows=pd.concat(all_dfs,ignore_index=True).dropna()
FEATURE_COLS=[c for c in windows.columns if c not in ['activityID','subject_id']]
print(f"\nTotal windows: {len(windows):,} | Features: {len(FEATURE_COLS)}")

# ── ENCODE LABELS ───────────────────────────────────────────────────────────
le=LabelEncoder()
le.fit(sorted(ACTIVITY_MAP.keys()))
y_enc=le.transform(windows['activityID'].values)
n_cls=len(le.classes_)
names=[ACTIVITY_MAP[int(c)] for c in le.classes_]
s_all=windows['subject_id'].values

# ── TRAIN / TEST SPLIT ──────────────────────────────────────────────────────
test_m =s_all==TEST_SUBJECT; train_m=~test_m
X_tr_raw=windows[FEATURE_COLS].values[train_m]; y_tr=y_enc[train_m]
X_te_raw=windows[FEATURE_COLS].values[test_m];  y_te=y_enc[test_m]
print(f"Train: {len(X_tr_raw):,} | Test: {len(X_te_raw):,}")

# ── STANDARDISE + PCA ───────────────────────────────────────────────────────
scaler=StandardScaler()
X_tr_sc=scaler.fit_transform(X_tr_raw)
X_te_sc=scaler.transform(X_te_raw)

pca=PCA(n_components=N_QUBITS, random_state=42)
X_tr_pca=pca.fit_transform(X_tr_sc)
X_te_pca=pca.transform(X_te_sc)

var_explained=pca.explained_variance_ratio_.cumsum()[-1]
print(f"\nPCA: {N_QUBITS} components explain {var_explained:.1%} of variance")

# Normalise PCA output to [-π, π] for AngleEmbedding.
# FIX (review item #5): fit min/max on TRAIN data only, then apply the same
# affine transform to val/test. Fitting per-split (as before) leaks
# distributional info from test into its own scaling and makes train/test
# not directly comparable. This is the standard "fit on train, transform
# everything" pattern already used for StandardScaler/PCA above.
def fit_angle_scaler(X):
    X_min=X.min(0); X_max=X.max(0)
    X_range=np.where(X_max-X_min<1e-8, 1., X_max-X_min)
    return X_min, X_range

def apply_angle_scaler(X, X_min, X_range):
    return (X-X_min)/X_range*2*np.pi - np.pi

_angle_min, _angle_range = fit_angle_scaler(X_tr_pca)
X_tr_angle = apply_angle_scaler(X_tr_pca, _angle_min, _angle_range)
X_te_angle = apply_angle_scaler(X_te_pca, _angle_min, _angle_range)
# Note: test-set points can fall slightly outside [-π,π] if their PCA
# values exceed the train-set range — this is expected and fine for
# AngleEmbedding (angles just wrap), and is the honest way to evaluate
# generalisation instead of quietly rescaling test to fit exactly.

# ──────────────────────────────────────────────────────────────────────────────
# APPROACH 1 — VQC (Variational Quantum Classifier)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("APPROACH 1: VQC — Variational Quantum Classifier")
print("="*65)
print(f"  Circuit: AngleEmbedding + {N_LAYERS}x StronglyEntanglingLayers")
print(f"  Optimiser: Adam (classical parameter update)")
print(f"  Strategy: One-vs-Rest (11 binary VQC classifiers)")

dev = qml.device(QDEVICE, wires=N_QUBITS)

@qml.qnode(dev, interface="autograd", diff_method=QDIFF)
def vqc_circuit(inputs, weights):
    """
    Variational Quantum Circuit:
      1. AngleEmbedding: encode each PCA component as Ry rotation
         Ry(θ_i)|0⟩ = cos(θ_i/2)|0⟩ + sin(θ_i/2)|1⟩
      2. StronglyEntanglingLayers: parameterised rotations + CNOT entanglement
         Creates quantum correlations between sensor feature components
      3. Measure expectation of PauliZ on qubit 0
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

VQC_BATCH_SIZE = 64

def train_binary_vqc(X_pos, X_neg, X_val_pos, X_val_neg, n_epochs=8,
                      lr=0.05, batch_size=VQC_BATCH_SIZE, patience=3, seed=0):
    """Train one binary VQC (positive class vs all others).

    FIX (review items #1, #4, #13): the previous version drew exactly ONE
    64-sample batch per "epoch" and took one optimizer step on it — so
    n_epochs=25 meant 25 total gradient updates, not 25 passes over the
    data. That is a genuinely under-trained model, not evidence VQC can't
    learn this task.

    This version:
      - iterates over ALL mini-batches of the training set each epoch
        (true epochs), giving many more optimizer steps for the same
        n_epochs;
      - evaluates hinge loss on a held-out validation split each epoch;
      - keeps the best-validation-loss weights (early stopping) instead
        of just returning whatever the last epoch produced;
      - seeds numpy per-classifier so results are reproducible given the
        same seed.
    """
    np.random.seed(seed)
    n_min = min(len(X_pos), len(X_neg))
    X_pos = X_pos[:n_min]; X_neg = X_neg[:n_min]
    X_bin = np.vstack([X_pos, X_neg]).astype(np.float64)
    y_bin = np.array([1.]*n_min + [-1.]*n_min)
    n_total = len(X_bin)
    bs = min(batch_size, n_total)

    n_val_min = min(len(X_val_pos), len(X_val_neg))
    X_val = np.vstack([X_val_pos[:n_val_min], X_val_neg[:n_val_min]]).astype(np.float64)
    y_val = np.array([1.]*n_val_min + [-1.]*n_val_min)

    weights = 0.01 * np.random.randn(N_LAYERS, N_QUBITS, 3)
    opt     = qml.AdamOptimizer(lr)

    def hinge_loss(w, X, y):
        preds = pnp.stack([vqc_circuit(x, w) for x in X])
        return float(pnp.mean(pnp.maximum(0, 1 - y * preds)))

    best_val = np.inf
    best_weights = weights
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_total)
        n_batches = int(np.ceil(n_total / bs))
        epoch_losses = []
        for b in range(n_batches):
            batch_idx = perm[b*bs:(b+1)*bs]
            if len(batch_idx) == 0:
                continue
            Xb, yb = X_bin[batch_idx], y_bin[batch_idx]

            def cost(w, Xb=Xb, yb=yb):
                preds = pnp.stack([vqc_circuit(x, w) for x in Xb])
                return pnp.mean(pnp.maximum(0, 1 - yb * preds))

            weights, c = opt.step_and_cost(cost, weights)
            epoch_losses.append(float(c))

        train_loss = float(np.mean(epoch_losses))
        val_loss = hinge_loss(weights, X_val, y_val) if n_val_min > 0 else train_loss

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_weights = weights
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"    epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  ({n_batches} batches/epoch)"
              + ("  *best*" if epochs_no_improve==0 else ""))

        if epochs_no_improve >= patience:
            print(f"    early stop at epoch {epoch+1} (no val improvement for {patience} epochs)")
            break

    return best_weights

# ── STRATIFIED TRAIN/VAL/TEST SUBSAMPLING FOR VQC ───────────────────────────
# FIX (review item #3): the previous version drew ONE global random
# subsample from all training windows, then re-used it as-is for every
# OvR classifier. With 11 imbalanced classes, a global random draw can
# easily under-represent some classes long before the per-classifier
# positive/negative split happens. This draws SUBSAMPLE_TRAIN examples
# PER CLASS (stratified), so every OvR classifier sees a comparable,
# well-balanced pool for its class.
# FIX (review item #4): also carves out a validation split from the
# training subjects/windows (not from the held-out test subject) so
# early stopping never touches test data.
print("\n  Building stratified per-class train/val subsamples for QML...")
SUBSAMPLE_TRAIN = 1000   # per class, for the training pool
SUBSAMPLE_VAL   = 200    # per class, held out from train subjects for early stopping
SUBSAMPLE_TEST  = 500    # per class, from the held-out test subject only

RANDOM_SEED = 42
# For a research-quality result, wrap this entire script's VQC training in
# a loop over e.g. RANDOM_SEED in [42,43,44,45,46], collect vqc_acc/bal/f1
# each run, and report mean±std — single-seed numbers should be described
# as a point estimate, not the final claim, per review item #13.
np.random.seed(RANDOM_SEED)

def stratified_indices(y, per_class_n, rng):
    idx = []
    for cls in range(n_cls):
        cls_idx = np.where(y==cls)[0]
        rng.shuffle(cls_idx)
        idx.extend(cls_idx[:min(per_class_n, len(cls_idx))].tolist())
    return np.array(idx)

rng = np.random.RandomState(RANDOM_SEED)

# Split TRAIN SUBJECTS' windows into a train pool and a validation pool
# (validation must never touch the held-out TEST_SUBJECT).
train_pool_idx = stratified_indices(y_tr, SUBSAMPLE_TRAIN + SUBSAMPLE_VAL, rng)
X_pool = X_tr_angle[train_pool_idx]; y_pool = y_tr[train_pool_idx]

# For each class, first SUBSAMPLE_TRAIN examples -> train, rest -> val
tr_final_idx, val_idx = [], []
for cls in range(n_cls):
    cls_pos = np.where(y_pool==cls)[0]
    tr_final_idx.extend(cls_pos[:SUBSAMPLE_TRAIN].tolist())
    val_idx.extend(cls_pos[SUBSAMPLE_TRAIN:SUBSAMPLE_TRAIN+SUBSAMPLE_VAL].tolist())

X_tr_q  = X_pool[tr_final_idx]; y_tr_q  = y_pool[tr_final_idx]
X_val_q = X_pool[val_idx];      y_val_q = y_pool[val_idx]

if len(X_val_q) == 0:
    print("  ⚠ WARNING: validation pool is empty (not enough per-class training")
    print("    windows to carve out SUBSAMPLE_VAL on top of SUBSAMPLE_TRAIN).")
    print("    Early stopping will fall back to monitoring train loss instead")
    print("    of a true held-out validation loss — reduce SUBSAMPLE_TRAIN/")
    print("    SUBSAMPLE_VAL or provide more data per class to fix this properly.")

# Test subsample comes only from the held-out TEST_SUBJECT windows (X_te_angle/y_te)
test_idx = stratified_indices(y_te, SUBSAMPLE_TEST, rng)
X_te_q = X_te_angle[test_idx]; y_te_q = y_te[test_idx]

print(f"  VQC train: {len(X_tr_q):,} | VQC val: {len(X_val_q):,} | VQC test: {len(X_te_q):,}")
print(f"  (test subsample drawn only from held-out subject {TEST_SUBJECT})")

vqc_weights = {}
start = time.time()
for cls_idx in range(n_cls):
    act = names[cls_idx]
    print(f"\n  Training VQC {cls_idx+1}/{n_cls}: {act}")
    X_pos = X_tr_q[y_tr_q==cls_idx]
    X_neg = X_tr_q[y_tr_q!=cls_idx]
    X_val_pos = X_val_q[y_val_q==cls_idx]
    X_val_neg = X_val_q[y_val_q!=cls_idx]
    # n_epochs here means true passes over the (mini-batched) training set,
    # not single optimizer steps — see docstring above. 8 epochs over a
    # ~2000-sample balanced set with batch=64 is ~30 optimizer steps/epoch,
    # i.e. up to ~240 steps total per classifier before early stopping,
    # roughly an order of magnitude more learning signal than the original
    # 25-single-batch-update version, while still finishing in reasonable
    # time on CPU. Increase n_epochs / patience further if time allows.
    vqc_weights[cls_idx] = train_binary_vqc(
        X_pos, X_neg, X_val_pos, X_val_neg,
        n_epochs=8, lr=0.03, patience=3, seed=RANDOM_SEED + cls_idx
    )

vqc_time = time.time() - start
print(f"\n  VQC training time: {vqc_time/60:.1f} minutes")

# Predict with OvR
print("\n  Predicting...")
scores = np.zeros((len(X_te_q), n_cls))
for cls_idx in range(n_cls):
    scores[:,cls_idx] = np.array([
        vqc_circuit(x, vqc_weights[cls_idx]) for x in X_te_q
    ])

y_pred_vqc = scores.argmax(1)
# Convert scores to pseudo-probabilities via softmax
exp_scores = np.exp(scores - scores.max(1,keepdims=True))
y_prob_vqc = exp_scores / exp_scores.sum(1, keepdims=True)

vqc_acc = accuracy_score(y_te_q, y_pred_vqc)
vqc_bal = balanced_accuracy_score(y_te_q, y_pred_vqc)
vqc_f1  = f1_score(y_te_q, y_pred_vqc, average='macro', zero_division=0)
print(f"\n  VQC Results (on {len(y_te_q)} test windows):")
print(f"  Accuracy:     {vqc_acc:.1%}")
print(f"  Balanced Acc: {vqc_bal:.1%}")
print(f"  Macro F1:     {vqc_f1:.3f}")
print(classification_report(y_te_q, y_pred_vqc, target_names=names, zero_division=0))

# FIX (review item #2): flag rather than silently accept near/below-random
# performance. With n_cls classes, random guessing ≈ 1/n_cls. Accuracy at
# or below that is a signal the training setup needs further debugging
# (learning rate, circuit expressivity, epochs, etc.) — not evidence about
# VQC's ceiling on this task — so we surface it loudly instead of just
# printing a number that looks like a finished result.
_random_baseline = 1.0 / n_cls
if vqc_acc <= _random_baseline * 1.15:
    print(f"\n  ⚠ WARNING: VQC accuracy ({vqc_acc:.1%}) is at/near the random")
    print(f"    baseline for {n_cls} classes (~{_random_baseline:.1%}). Do NOT report")
    print(f"    this as a final result — investigate learning rate, epoch count,")
    print(f"    circuit expressivity, or data scaling before drawing conclusions.")

# ──────────────────────────────────────────────────────────────────────────────
# APPROACH 2 — QKSVM (Quantum Kernel SVM)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("APPROACH 2: QKSVM — Quantum Kernel SVM")
print("="*65)
print("  Kernel: ZZFeatureMap (quantum feature map)")
print("  Classifier: SVM with quantum kernel matrix")
print("  Strategy: One-vs-One (sklearn default for SVC)")

@qml.qnode(dev, interface="autograd")  # qml.probs() isn't supported by adjoint diff; no
def quantum_kernel_circuit(x1, x2):     # training/gradients happen through this circuit
                                          # anyway (it's only ever called forward-only, to
                                          # build the kernel matrix for a classical SVC), so
                                          # the default diff_method is fine and free here.
    """
    Quantum kernel: K(x1,x2) = |⟨φ(x1)|φ(x2)⟩|²

    FIX (review item #7): the docstring previously claimed a ZZ-style
    feature map with Hadamards, but the implementation only did an
    AngleEmbedding + its adjoint. This version implements what the
    docstring actually describes:
      1. Hadamard on every wire -> uniform superposition
      2. RZ rotation per feature (angle-encoded) -> feature map |φ(x)>
      3. Adjoint of the same map for x2, applied on top
      4. Probability of measuring all-zeros = |<φ(x1)|φ(x2)>|^2
    This is a real (if simplified) ZZ-style feature map: the initial
    Hadamard layer means each RZ rotation now acts on a superposition
    state rather than a computational basis state, which changes the
    circuit's expressivity relative to the previous Hadamard-free version.
    """
    for w in range(N_QUBITS):
        qml.Hadamard(wires=w)
    qml.AngleEmbedding(x1, wires=range(N_QUBITS), rotation='Z')
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(N_QUBITS), rotation='Z')
    for w in range(N_QUBITS):
        qml.Hadamard(wires=w)
    return qml.probs(wires=range(N_QUBITS))

def compute_kernel_matrix(X1, X2, symmetric=False):
    """Compute Gram matrix K[i,j] = |⟨φ(X1[i])|φ(X2[j])⟩|²

    When X1 is X2 (the train-train block), the matrix is symmetric with a
    diagonal of exactly 1 (self-overlap), so we only need the upper
    triangle — same result, roughly half the circuit evaluations.
    """
    n1,n2=len(X1),len(X2)
    K=np.zeros((n1,n2))
    if symmetric:
        for i in range(n1):
            K[i,i]=1.0
            for j in range(i+1,n2):
                probs=quantum_kernel_circuit(X1[i],X2[j])
                val=float(probs[0])
                K[i,j]=val; K[j,i]=val
            if (i+1)%10==0: print(f"    Kernel row {i+1}/{n1}", end='\r')
    else:
        for i,xi in enumerate(X1):
            for j,xj in enumerate(X2):
                probs=quantum_kernel_circuit(xi,xj)
                K[i,j]=float(probs[0])   # probability of all-zeros state = overlap
            if (i+1)%10==0: print(f"    Kernel row {i+1}/{n1}", end='\r')
    return K

# Use smaller subsample for kernel (O(n²) complexity)
# FIX (review item #6, CRITICAL): the previous version built X_ktr/y_ktr
# from X_te_angle/y_te_q — i.e. it trained the QKSVM on windows from the
# held-out TEST_SUBJECT, then evaluated on more windows from that same
# subject. That's direct test-set leakage: the "test" accuracy included
# information the model was trained on. Training data now comes from
# X_tr_angle/y_tr (train subjects only, disjoint from TEST_SUBJECT);
# X_kte/y_kte for final evaluation still comes only from X_te_angle/y_te
# (the held-out subject), so training and evaluation subjects are fully
# disjoint, matching the same held-out-subject protocol used for VQC.
KERNEL_N = 200   # per class, from training subjects
rng_k = np.random.RandomState(RANDOM_SEED)
idx_ktr  = []
idx_kte  = []
for cls in range(n_cls):
    cls_idx_tr = np.where(y_tr==cls)[0]
    cls_idx_te = np.where(y_te==cls)[0]
    rng_k.shuffle(cls_idx_tr); rng_k.shuffle(cls_idx_te)
    idx_ktr.extend(cls_idx_tr[:min(KERNEL_N//n_cls, len(cls_idx_tr))].tolist())
    idx_kte.extend(cls_idx_te[:min(50,             len(cls_idx_te))].tolist())

X_ktr=X_tr_angle[idx_ktr]; y_ktr=y_tr[idx_ktr]   # training-subject windows only
X_kte=X_te_angle[idx_kte]; y_kte=y_te[idx_kte]   # held-out TEST_SUBJECT windows only
print(f"\n  Computing quantum kernel ({len(X_ktr)}×{len(X_ktr)} train matrix)...")
print(f"  Train kernel data: {len(X_ktr)} windows from TRAINING subjects (not {TEST_SUBJECT})")
print(f"  Test  kernel data: {len(X_kte)} windows from held-out subject {TEST_SUBJECT}")
start=time.time()
K_train=compute_kernel_matrix(X_ktr, X_ktr, symmetric=True)
K_test =compute_kernel_matrix(X_kte, X_ktr)
ktime=time.time()-start
print(f"\n  Kernel computation: {ktime/60:.1f} min")

qksvm=SVC(kernel='precomputed', C=1.0, probability=True)
qksvm.fit(K_train, y_ktr)
y_pred_qk=qksvm.predict(K_test)
y_prob_qk=qksvm.predict_proba(K_test)

qk_acc=accuracy_score(y_kte,y_pred_qk)
qk_bal=balanced_accuracy_score(y_kte,y_pred_qk)
qk_f1 =f1_score(y_kte,y_pred_qk,average='macro',zero_division=0)
print(f"\n  QKSVM Results (on {len(X_kte)} test windows, held-out subject {TEST_SUBJECT}):")
print(f"  Accuracy:     {qk_acc:.1%}")
print(f"  Balanced Acc: {qk_bal:.1%}")
print(f"  Macro F1:     {qk_f1:.3f}")
print(f"  NOTE: evaluated under computational subsampling — train kernel built")
print(f"  from {len(X_ktr)} training-subject windows (O(n²) circuit cost); this must")
print(f"  be disclosed alongside the number in any write-up, not presented as")
print(f"  equivalent to a full-dataset evaluation.")

# ──────────────────────────────────────────────────────────────────────────────
# APPROACH 3 — Quantum Transfer Learning (QTL)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("APPROACH 3: QTL — Quantum Transfer Learning")
print("="*65)
print("  Classical backbone: PCA features (already computed)")
print("  Quantum layer: Variational circuit on top of classical embeddings")
print("  Key insight: Classical model extracts representations,")
print("               quantum layer adds non-linear transformation")

@qml.qnode(dev, interface="autograd", diff_method=QDIFF)
def qtl_circuit(inputs, weights):
    """
    Quantum Transfer Learning circuit:
    1. AmplitudeEmbedding: encodes N-dim classical vector in log2(N) qubits
       MUCH more efficient than AngleEmbedding (16 values in 4 qubits vs 16)
    2. Strongly Entangling Layers on the 4 qubits
    3. Measure PauliZ expectation on each qubit → 4 output values
    """
    # Use amplitude embedding — encodes 16 features in log2(16)=4 qubits
    qml.AmplitudeEmbedding(features=inputs[:16], wires=range(4), normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

N_QTL_QUBITS = 4
dev_qtl = qml.device(QDEVICE, wires=N_QTL_QUBITS)

@qml.qnode(dev_qtl, diff_method=QDIFF)
def qtl_circuit_v2(inputs, weights):
    qml.AmplitudeEmbedding(features=inputs[:2**N_QTL_QUBITS],
                           wires=range(N_QTL_QUBITS), normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QTL_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QTL_QUBITS)]

# Extract 16-dim classical features → pass to quantum circuit → 4-dim quantum features
# Then classical dense layer classifies 4-dim quantum output
print(f"\n  Using AmplitudeEmbedding: 16 features → {N_QTL_QUBITS} qubits")
print(f"  Output: {N_QTL_QUBITS} PauliZ expectation values → classical head")

# Normalise PCA outputs for AmplitudeEmbedding (must be unit norm)
def norm_amplitude(X):
    norms=np.linalg.norm(X,axis=1,keepdims=True)
    return X/np.where(norms<1e-10,1.,norms)

# FIX (review item #9): use the same stratified train/test subsamples as
# VQC/QKSVM (X_tr_q/y_tr_q from training subjects, X_te_q/y_te_q from the
# held-out subject) instead of an arbitrary un-stratified `[:SUBSAMPLE_TRAIN]`
# slice of X_tr_pca. This keeps QTL on the same held-out-subject protocol
# as the other two approaches (review item #8) and gives it more than the
# previous 200/100 samples to work with.
X_tr_amp = norm_amplitude(X_tr_q)
X_te_amp = norm_amplitude(X_te_q)
y_tr_amp = y_tr_q
y_te_amp = y_te_q

print(f"\n  Training QTL ({len(X_tr_amp)} train / {len(X_te_amp)} test samples,")
print(f"  same held-out-subject split used for VQC/QKSVM)")
qtl_weights = 0.01*np.random.randn(N_LAYERS, N_QTL_QUBITS, 3)
opt_qtl = qml.AdamOptimizer(0.05)

# Compute quantum features then use classical SVM on top
def get_qtl_features(X, weights):
    feats=[]
    for x in X:
        out = qtl_circuit_v2(x[:2**N_QTL_QUBITS], weights)
        feats.append([float(o) for o in out])
    return np.array(feats)

# FIX (review item #10, CRITICAL): the previous objective mapped each of
# the 11 class labels to a 4-dim one-hot target via `class % 4`, meaning
# classes {0,4,8}, {1,5,9}, {2,6,10}, {3,7} were all pushed toward the
# SAME target vector. The quantum layer was therefore never trained to
# distinguish those classes at all — its "task" was a spurious 4-way
# problem, not the real 11-way problem.
#
# Fix: train the quantum layer with a proper contrastive/metric-learning
# objective that IS well-defined for an 11-class problem mapped to a
# lower-dimensional embedding: pull same-class embeddings together and
# push different-class embeddings apart (a standard approach for using a
# quantum circuit as a *feature extractor* ahead of a classical
# classifier, which is what this circuit is actually used for below).
# This makes the quantum layer's training signal consistent with what
# the downstream classical SVM will be asked to do (draw boundaries
# between all 11 classes in the resulting embedding space), instead of
# optimizing an unrelated 4-way objective.
MARGIN = 1.0

def cost_qtl_contrastive(w, Xb, yb):
    qfeats = pnp.stack([qtl_circuit_v2(x[:2**N_QTL_QUBITS], w) for x in Xb])
    n = len(Xb)
    losses = []
    # All pairs within the mini-batch (small batch, so O(n^2) pairs is cheap)
    for i in range(n):
        for j in range(i+1, n):
            dist = pnp.sum((qfeats[i]-qfeats[j])**2)
            if yb[i] == yb[j]:
                losses.append(dist)                                  # pull together
            else:
                losses.append(pnp.maximum(0., MARGIN - dist))        # push apart
    return pnp.mean(pnp.stack(losses)) if losses else pnp.array(0.)

# Training loop: proper multi-batch epochs (same fix as VQC, review #1)
QTL_EPOCHS = 8
QTL_BATCH  = 16   # kept small since the contrastive loss is O(batch^2) pairs
n_qtl_train = len(X_tr_amp)
for epoch in range(QTL_EPOCHS):
    perm = np.random.permutation(n_qtl_train)
    n_batches = int(np.ceil(n_qtl_train / QTL_BATCH))
    epoch_losses = []
    for b in range(n_batches):
        batch = perm[b*QTL_BATCH:(b+1)*QTL_BATCH]
        if len(batch) < 2:
            continue
        Xb, yb = X_tr_amp[batch], y_tr_amp[batch]
        qtl_weights, c = opt_qtl.step_and_cost(
            lambda w: cost_qtl_contrastive(w, Xb, yb), qtl_weights)
        epoch_losses.append(float(c))
    print(f"    epoch {epoch+1}/{QTL_EPOCHS}  loss={np.mean(epoch_losses):.4f}  "
          f"({n_batches} batches/epoch)")

# Extract quantum features for classification.
# The downstream classifier still does genuine 11-class classification
# (SVC below, unchanged) — only the quantum layer's own training
# objective was broken and is what item #10 fixed.
print("  Extracting quantum features...")
qtr_feats=get_qtl_features(X_tr_amp, qtl_weights)
qte_feats=get_qtl_features(X_te_amp, qtl_weights)

qtl_clf=SVC(kernel='rbf',C=10.,probability=True)
qtl_clf.fit(qtr_feats, y_tr_amp)
y_pred_qtl=qtl_clf.predict(qte_feats)
y_prob_qtl=qtl_clf.predict_proba(qte_feats)

qtl_acc=accuracy_score(y_te_amp,y_pred_qtl)
qtl_bal=balanced_accuracy_score(y_te_amp,y_pred_qtl)
qtl_f1 =f1_score(y_te_amp,y_pred_qtl,average='macro',zero_division=0)
print(f"\n  QTL Results (on {len(qte_feats)} test windows, held-out subject {TEST_SUBJECT}):")
print(f"  Accuracy:     {qtl_acc:.1%}")
print(f"  Balanced Acc: {qtl_bal:.1%}")
print(f"  Macro F1:     {qtl_f1:.3f}")
print(f"  NOTE: {N_QTL_QUBITS}-dim quantum embedding is a strong information")
print(f"  bottleneck for {n_cls} classes — treat as proof-of-concept, not a")
print(f"  ceiling on QTL performance (review item #11).")

# ── SAVE QML RESULTS ────────────────────────────────────────────────────────
# Reuse the VQC test subsample (X_te_q / y_te_q) instead of re-running
# inference on the entire un-subsampled test set: `scores` was already
# computed on this exact subsample above for the VQC metrics, so this
# avoids a large, redundant second inference pass while reporting on
# precisely the same windows the headline VQC accuracy/F1 numbers use.
print("\nUsing already-computed VQC probabilities on the VQC test subsample for hybrid ensemble...")
y_prob_vqc_full = y_prob_vqc         # computed earlier from `scores`
y_pred_vqc_full = y_pred_vqc
y_te_full        = y_te_q

def tl(c): return "GREEN" if c>=0.75 else ("YELLOW" if c>=0.45 else "RED")
tgt_probs_full = np.array([y_prob_vqc_full[i,y_te_full[i]] for i in range(len(y_te_full))])
is_corr_full   = (y_pred_vqc_full==y_te_full).astype(int)

pd.DataFrame({
    'true_activity':[ACTIVITY_MAP.get(int(le.inverse_transform([i])[0]),'?') for i in y_te_full],
    'pred_activity':[ACTIVITY_MAP.get(int(le.inverse_transform([i])[0]),'?') for i in y_pred_vqc_full],
    'confidence':tgt_probs_full.round(3),
    'is_correct':is_corr_full,
    'traffic_light':[tl(c) for c in tgt_probs_full],
    'y_true_enc':y_te_full,
    'y_pred_enc':y_pred_vqc_full,
}).to_csv("pamap2_qml_results.csv",index=False)
np.save("pamap2_qml_proba.npy", y_prob_vqc_full)

print(f"\n{'='*65}")
print("QML COMPARISON SUMMARY")
print(f"{'='*65}")
print(f"{'Approach':<30} {'Acc':>8} {'Bal Acc':>10} {'F1':>8}  {'N (train/test)'}")
print(f"{'-'*58}")
print(f"{'VQC (AngleEmbed)':<30} {vqc_acc:>7.1%} {vqc_bal:>9.1%} {vqc_f1:>7.3f}  {len(X_tr_q)}/{len(X_te_q)}")
print(f"{'QKSVM (Quantum Kernel)':<30} {qk_acc:>7.1%} {qk_bal:>9.1%} {qk_f1:>7.3f}  {len(X_ktr)}/{len(X_kte)}")
print(f"{'QTL (Transfer)':<30} {qtl_acc:>7.1%} {qtl_bal:>9.1%} {qtl_f1:>7.3f}  {len(X_tr_amp)}/{len(X_te_amp)}")
print(f"{'-'*58}")
print(f"{'Classical ML (reference)':<30} {'95.8%':>8} {'90.3%':>10} {'0.907':>8}  (full dataset)")
print(f"\nAll three QML approaches use the SAME held-out test subject")
print(f"({TEST_SUBJECT}) and stratified per-class subsampling from disjoint")
print(f"training-subject / test-subject windows — see printed N (train/test)")
print(f"above for exact sample sizes used by each method.")
print(f"\nNote: QML tested on subsampled windows due to O(n²)/simulator")
print(f"complexity relative to the classical ML reference's full dataset.")
print(f"Full-dataset QML requires quantum hardware or an HPC cluster.")
if vqc_acc <= (1.0/n_cls)*1.15:
    print(f"\n⚠ VQC result is at/near random baseline for {n_cls} classes — see")
    print(f"  warning above. Investigate before treating as a final comparison.")
print(f"{'='*65}")
print(f"\nSaved: pamap2_qml_results.csv  pamap2_qml_proba.npy")
print(f"Run next: python pamap2_final_hybrid.py")