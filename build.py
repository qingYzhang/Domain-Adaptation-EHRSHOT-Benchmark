#!/usr/bin/env python3
"""
select_and_build_train_pkl.py

Select patients by average mapping score, tokenize their history, and save train.pkl
for fine-tuning Catch-FM.

Example:
python build.py \
    --patients_file ehrshot_in_nhird_patients_v800.json \
    --vocab_path vocabulary.json \
    --out_pkl train_0.8.pkl \
    --score_threshold 0.8 \
    --block_size 2048 \
    --default_label 0
"""

import argparse
import json
import pickle
from tqdm import tqdm
from pretokenize_code import history_to_ids_sequence


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patients_file", required=True, help="NHIRD-format patients file (JSONL or JSON list)")
    p.add_argument("--vocab_path", required=True, help="Path to vocabulary.json for code2idx mapping")
    p.add_argument("--out_pkl", default="train.pkl", help="Output pickle filename")
    p.add_argument("--score_threshold", type=float, default=0.8, help="Include patients with avg score >= threshold")
    p.add_argument("--label_field", default="/data/stevenz3/EHR/benchmark/new_pancan/labeled_patients.csv", help="Optional CSV file: patient_id,label")
    p.add_argument("--default_label", type=int, default=0, help="Default label if no CSV provided")
    p.add_argument("--block_size", type=int, default=2048, help="Maximum sequence length (truncate/pad)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def read_patients_file(path):
    patients = []
    with open(path, "r") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            patients = json.load(f)
        else:
            for line in f:
                if line.strip():
                    patients.append(json.loads(line))
    return patients


def load_labels_csv(csv_path):
    labels = {}
    with open(csv_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                pid = parts[0].strip()
                # print(type(pid))
                # if pid == '115973821':
                #     print(f"Loading label for patient {pid}: {parts}")
                try:
                    labels[pid] = 1 if parts[-2].strip().lower() in ["1", "true", "yes"] else 0
                    # print(type(pid))
                    if pid == '115973821':
                        print(f"Loaded label for patient {pid}: {labels[pid]} and {parts}")
                except ValueError:
                    labels[pid] = 0
    return labels


def average_safe(lst):
    return float(sum(lst)) / len(lst) if lst else None


def pad_or_trim(seq, max_len, pad_id=0):
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [pad_id] * (max_len - len(seq))


def build_examples(patients, vocab, score_threshold, label_map, default_label, block_size, verbose=False):
    examples = []
    skipped = 0
    # print(1)
    for patient in tqdm(patients, desc="Selecting patients"):
        pid = str(patient.get("patient_id", ""))
        history = patient.get("history", [])
        if not history:
            skipped += 1
            continue
        # print(2)
        # --- compute patient-level average of visit avg_score_all_codes ---
        total_weighted_score = 0.0
        total_codes = 0
        # print(3)
        for v in history:
            # print(v)
            avg_s = v.get("avg_score_all_codes")
            n_codes = v.get("n_total_codes", 0)
            # print(avg_s, n_codes)
            if avg_s is not None and n_codes > 0:
                total_weighted_score += avg_s * n_codes
                total_codes += n_codes

        patient_avg = total_weighted_score / total_codes if total_codes > 0 else None


        if patient_avg is None or patient_avg < score_threshold:
            skipped += 1
            print("below threshold")
            continue

        try:
            input_ids, positions = history_to_ids_sequence(history, vocab)
        except Exception as e:
            if verbose:
                tqdm.write(f"[WARN] Tokenization failed for patient {pid}: {e}")
            skipped += 1
            continue

        input_ids = pad_or_trim(input_ids, block_size, vocab["code2idx"].get("PAD", 0))
        positions = pad_or_trim(positions, block_size, 0)
        # print(label_map)
        # print(pid)
        # print(pid, pid in label_map)
        if pid not in label_map:
            print("not in label map")
            if verbose:
                tqdm.write(f"[WARN] No label for patient {pid}, skipping.")
            skipped += 1
            continue
        examples.append({
            "patient_id": pid,
            "input_ids": input_ids,
            "positions": positions,
            "label": int(label_map.get(pid)),
            "avg_score": float(patient_avg),
        })

    return examples, skipped


def main():
    args = parse_args()
    vocab = json.load(open(args.vocab_path))
    patients = read_patients_file(args.patients_file)
    label_map = load_labels_csv(args.label_field) if args.label_field else {}

    examples, skipped = build_examples(
        patients, vocab, args.score_threshold, label_map, args.default_label, args.block_size, args.verbose
    )

    print(f"✅ Selected {len(examples)} patients, skipped {skipped}.")
    if examples:
        with open(args.out_pkl, "wb") as f:
            pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"💾 Saved to {args.out_pkl}")


if __name__ == "__main__":
    main()
