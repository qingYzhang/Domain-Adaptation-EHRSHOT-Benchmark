import pickle
import json
import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    PowerTransformer, QuantileTransformer, PolynomialFeatures
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV, 
    PredefinedSplit
)


params = {
    "lr": {
        "C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000, 2e-5, 2e-4, 2e-3, 2e-2, 2e-1, 2, 20, 200, 2000, 20000, 200000, 3e-5, 3e-4, 3e-3, 3e-2, 3e-1, 3, 30, 300, 3000, 30000, 300000, 4e-5, 4e-4, 4e-3, 4e-2, 4e-1, 4, 40, 400, 4000, 40000, 400000, 5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 5, 50, 500, 5000, 50000, 500000, 6e-5, 6e-4, 6e-3, 6e-2, 6e-1, 6, 60, 600, 6000, 60000, 600000, 7e-5, 7e-4, 7e-3, 7e-2, 7e-1, 7, 70, 700, 7000, 70000, 700000, 8e-5, 8e-4, 8e-3, 8e-2, 8e-1, 8, 80, 800, 8000, 80000, 800000, 9e-5, 9e-4, 9e-3, 9e-2, 9e-1, 9, 90, 900, 9000, 90000, 900000],
        "penalty": [None, "l2", "l1"],
        "fit_intercept": [True, False],
        "solver": ["newton-cg", "lbfgs", "liblinear"],
    },
}

# Define scalers dictionary
scalers = {
    "MaxAbsScaler": MaxAbsScaler(),
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
    "None": None
}


def eval_metrics(total_preds_probs, total_targets):
    # Calculate evaluation metrics
    average_precision = average_precision_score(total_targets, total_preds_probs)
    roc_auc = roc_auc_score(total_targets, total_preds_probs)
    return roc_auc, average_precision

def apply_scaler(scaler, X):
    """Apply scaler to data, handling None case"""
    if scaler is None:
        return X
    return scaler.fit_transform(X)

def run_eval_k_shot(all_shot_datasets, k, scaler_name):
    scores = defaultdict(list)
    scaler = scalers[scaler_name]

    for repeat_num, shot_data in all_shot_datasets[k].items():
        # print(repeat_num, shot_data)
        X_train = apply_scaler(scaler, shot_data["train_k"]["features"])
        y_train = shot_data["train_k"]["labels"]
        X_val = apply_scaler(scaler, shot_data["val_k"]["features"])
        y_val = shot_data["val_k"]["labels"]
        X_test = apply_scaler(scaler, all_shot_datasets["test"]["features"])
        y_test = all_shot_datasets["test"]["labels"]
    
        X_train_val = np.vstack((X_train, X_test))
        y_train_val = np.concatenate((y_train, y_test))

        train_indices = -1 * np.ones(X_train.shape[0])
        val_indices = np.zeros(X_test.shape[0])
        split_indices = np.concatenate([train_indices, val_indices])
        predefined_split = PredefinedSplit(test_fold=split_indices)

        clf = LogisticRegression(max_iter=50000)
        grid_search = GridSearchCV(estimator=clf, param_grid=params["lr"], cv=predefined_split, scoring="roc_auc", n_jobs=args.n_jobs)
        
        # Calculate total number of parameter combinations
        total_combinations = 1
        for param_name, param_values in params["lr"].items():
            total_combinations *= len(param_values)
        print(f"Total parameter combinations to try: {total_combinations}")
        
        grid_search.fit(X_train_val, y_train_val)
        best_model = grid_search.best_estimator_
        print("Best params for shot {} with {}: {}".format(k, scaler_name, grid_search.best_params_))

        best_model.fit(X_train, y_train)
        probs = best_model.predict_proba(X_test)[:,1]
        
        roc_auc, average_precision = eval_metrics(probs, y_test) 
                
        scores["AUROC"].append(roc_auc)
        scores["AUPRC"].append(average_precision)

    averaged_scores = {metric: sum(values) / len(values) for metric, values in scores.items()}

    return averaged_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--task", type=str, required=True,
    #                     help="Task name to evaluate, e.g., 'new_lupus', 'new_pancan', etc.")
    parser.add_argument("--model_size", type=str, default="160m", required=False, choices=["160m", "1b", "2.4b"],
                        help="Model size to use for generating features.")
    parser.add_argument("--scaler", type=str, default="MaxAbsScaler", 
                        choices=["MaxAbsScaler", "StandardScaler", "MinMaxScaler", "RobustScaler", "None"],
                        help="Scaler to use for feature preprocessing.")
    parser.add_argument("--n_jobs", type=int, default=60,
                        help="Number of jobs to run in parallel for grid search.")
    args = parser.parse_args()
    
    # all_shot_scores = {}
    # all_shot_datasets = pickle.load(open(f"benchmark/{args.task}/all_shot_datasets_{args.model_size}.pkl", "rb"))


    benchmark_dir = "benchmark"
    # task_dirs = [d for d in os.listdir(benchmark_dir)
    #              if os.path.isdir(os.path.join(benchmark_dir, d))]
    
    task_dirs = [
        #'chexpert',
        # 'new_lupus',
        # 'guo_icu',
        # 'new_hypertension',
        # 'new_acutemi',
        # 'new_hyperlipidemia',
        # 'guo_los',
        # 'guo_readmission',
        # 'new_celiac',
        'new_pancan',
        # 'lab_hyponatremia',
        # 'lab_hyperkalemia',
        # 'lab_anemia',
        # 'lab_thrombocytopenia',
        # 'lab_hypoglycemia'
    ]

    for task in task_dirs:
        print(f"\nProcessing task: {task}")
        dataset_path = os.path.join(benchmark_dir, task, f"all_shot_datasets_{args.model_size}.pkl")
        # if not os.path.exists(dataset_path) or task == "lab_hyperkalemia" or task == "lab_hypoglycemia" or task == "lab_thrombocytopenia":
        if not os.path.exists(dataset_path):
            print(f"Dataset not found for task {task}, skipping.")
            continue

        all_shot_scores = {}
        all_shot_datasets = pickle.load(open(dataset_path, "rb"))

        for k in all_shot_datasets:
            if k == "test":
                continue
            print(f"Evaluating k={k} with {args.scaler}...")
            averaged_scores = run_eval_k_shot(all_shot_datasets, k, args.scaler)
            print(json.dumps(averaged_scores, indent=4))
            all_shot_scores[k] = averaged_scores

        output_path = os.path.join(benchmark_dir, task, f"all_shot_scores_{args.model_size}_{args.scaler}.json")
        with open(output_path, "w") as f:
            json.dump(all_shot_scores, f, indent=4)

        print(f"Saved results to {output_path}")


    # for k in all_shot_datasets:
    #     if k == "test" or k == "-1":
    #         continue
    #     print(f"Evaluating k={k} with {args.scaler}...")
    #     averaged_scores = run_eval_k_shot(all_shot_datasets, k, args.scaler)
    #     print(json.dumps(averaged_scores, indent=4))
    #     all_shot_scores[k] = averaged_scores

    # # Add scaler information to the output filename
    # output_filename = f"benchmark/{args.task}/all_shot_scores_{args.model_size}_{args.scaler}.json"
    # with open(output_filename, "w") as f:
    #     json.dump(all_shot_scores, f, indent=4)
    
    # print(f"Results saved to: {output_filename}")