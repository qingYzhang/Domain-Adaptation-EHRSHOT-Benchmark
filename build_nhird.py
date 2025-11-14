import pickle, json
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed

# ======= CONFIG =======
STANDARD_PATH = "../data/pickle/train_0.8.pkl"
TRAIN_PATH = "/data/user_data/stevenz3/nhird_data/valid.pkl"
VOCAB_PATH = "vocabulary.json"    # your JSON example
THRESHOLD = 0.35          # adjust based on histogram
OUTPUT_PATH = "valid.pkl"
USE_TF = True                # True → term-frequency weighting; False → binary
N_JOBS = 8           # tune based on HPC cores
BATCH_SIZE = 1000    # avoids memory blow-up

# ===== LOAD =====
with open(STANDARD_PATH, "rb") as f:
    standard_patients = pickle.load(f)
with open(TRAIN_PATH, "rb") as f:
    train_patients = pickle.load(f)
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

code2idx = vocab["code2idx"]
vocab_size = len(code2idx)
pad_idx = code2idx.get("PAD", 0)

# ===== PATIENT → SPARSE VECTOR =====
def patient_to_sparse(patient):
    ids = np.array(patient["input_ids"], dtype=int)
    ids = ids[ids != pad_idx]
    if len(ids) == 0:
        return [], []
    unique, counts = np.unique(ids, return_counts=True)
    if USE_TF:
        data = counts / counts.sum()
    else:
        data = np.ones_like(unique, dtype=np.float32)
    return unique, data

# Build sparse matrices
def build_sparse_matrix(patients):
    row_idx, col_idx, data = [], [], []
    for i, p in enumerate(patients):
        cols, vals = patient_to_sparse(p)
        row_idx.extend([i]*len(cols))
        col_idx.extend(cols)
        data.extend(vals)
    mat = csr_matrix((data, (row_idx, col_idx)), shape=(len(patients), vocab_size), dtype=np.float32)
    return normalize(mat)  # L2 normalize rows

print("Building sparse matrices...")
standard_mat = build_sparse_matrix(standard_patients)
train_mat = build_sparse_matrix(train_patients)

# ===== COSINE SIMILARITY IN BATCHES =====
def process_batch(start):
    end = min(start + BATCH_SIZE, train_mat.shape[0])
    sims = train_mat[start:end] @ standard_mat.T
    # Convert to dense 1D float array
    max_sims = sims.max(axis=1).toarray().ravel()
    selected = [
        train_patients[start + i]
        for i, s in enumerate(max_sims)
        if float(s) >= THRESHOLD
    ]
    return selected, max_sims.tolist()


print("Computing batched similarities...")
results = Parallel(n_jobs=N_JOBS)(
    delayed(process_batch)(i)
    for i in tqdm(range(0, train_mat.shape[0], BATCH_SIZE))
)

# Merge results
selected_patients = []
similarities = []
for sel, sims in results:
    selected_patients.extend(sel)
    similarities.extend(sims)

# ===== SAVE =====
OUTPUT_PATH = OUTPUT_PATH.replace(".pkl", f"_th{THRESHOLD}_num{len(selected_patients)}.pkl")
# create a directory for this threshold (use safe name) and ensure it exists
out_dir = f"th{str(THRESHOLD).replace('.', '_')}"
os.makedirs(out_dir, exist_ok=True)
OUTPUT_PATH = os.path.join(out_dir, OUTPUT_PATH)
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(selected_patients, f)

print(f"✅ Selected {len(selected_patients)} / {len(train_patients)} (threshold={THRESHOLD})")

# ======= OPTIONAL: visualize similarity distribution =======
import matplotlib.pyplot as plt
plt.hist(similarities, bins=50)
plt.xlabel("Max cosine similarity to standard patients")
plt.ylabel("Patient count")
plt.title("Code-based similarity distribution")
plt.show()
plt.savefig("similarity_histogram.png")