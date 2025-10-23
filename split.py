#!/usr/bin/env python3
"""
split_train_valid_pkl.py
Randomly split patients into train.pkl and valid.pkl
"""

"""
Usage:
python split.py \
    --input_pkl train_0.8.pkl \
    --train_out ./pickle/0.8/train.pkl \
    --valid_out ./pickle/0.8/valid.pkl \
    --valid_ratio 0.1
"""
import argparse
import pickle
import random
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_pkl", required=True, help="Input train.pkl file (full set)")
    p.add_argument("--train_out", default="train.pkl")
    p.add_argument("--valid_out", default="valid.pkl")
    p.add_argument("--valid_ratio", type=float, default=0.1, help="Fraction for validation set")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    with open(args.input_pkl, "rb") as f:
        data = pickle.load(f)

    random.shuffle(data)
    n_valid = int(len(data) * args.valid_ratio)
    valid = data[:n_valid]
    train = data[n_valid:]

    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    with open(args.train_out, "wb") as f:
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.valid_out, "wb") as f:
        pickle.dump(valid, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Train size: {len(train)} | Valid size: {len(valid)}")
    print(f"💾 Saved {args.train_out} and {args.valid_out}")


if __name__ == "__main__":
    main()
