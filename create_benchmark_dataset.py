import json
import csv
import os
import datetime
import pickle
import argparse
from tqdm import tqdm
from get_embedding import generate_all_features, load_model
import torch
from joblib import Parallel, delayed


def parse_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

def csv_to_dict(filename):
    data = []

    with open(filename, "r") as file:
        csv_read_file = csv.DictReader(file)
        for row in csv_read_file:
            data.append(row)

    return data

def load_patients(filename):
    patients = {}

    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                patient = json.loads(line.strip())
                patients[patient["patient_id"]] = patient
    return patients 

def get_prediction_time_index(prediction_time, time_index):
    prediction_time = parse_date(prediction_time)

    for visit_time, index in time_index.items():
        visit_time = parse_date(visit_time)
        if visit_time > prediction_time:
            return index - 1

    return index - 1

def get_single_data(patient, prediction_time, label):
    prediction_index = get_prediction_time_index(prediction_time, patient["time_index"])

    return {
        "patient_id": patient["patient_id"],
        "history": patient["history"][:prediction_index + 1],
        "label": 1 if label == "True" else 0 
    }

def load_labeled_dataset(patients, task_path, split_path, model, vocab, batch_size=32):
    label_infos = csv_to_dict(os.path.join(task_path, "labeled_patients.csv"))
    split_infos = dict([(split["omop_person_id"], split["split"]) for split in csv_to_dict(split_path)])

    train_dataset= {}
    val_dataset = {}
    test_dataset = {}

    print(f"Processing {len(label_infos)} labeled patients...")
    for label_info in tqdm(label_infos, desc="Processing labeled patients"):
        patient_id = label_info["patient_id"]
        prediction_time = label_info["prediction_time"]
        label = label_info["value"]
        split = split_infos[patient_id]
        data = get_single_data(patients[patient_id], prediction_time, label)
        prediction_time = prediction_time.replace(" ", "T")

        if split == "train":
            train_dataset[(patient_id, prediction_time)] = data
        elif split == "val":
            val_dataset[(patient_id, prediction_time)] = data
        elif split == "test":
            test_dataset[(patient_id, prediction_time)] = data

    print(f"Creating maps and data lists...")
    train_map = {}
    train_data = []
    for idx, (k, data) in enumerate(train_dataset.items()):
        train_map[k] = idx
        train_data.append(data)

    val_map = {}
    val_data = []
    for idx, (k, data) in enumerate(val_dataset.items()):
        val_map[k] = idx
        val_data.append(data)
    
    test_map = {}
    test_data = []
    for idx, (k, data) in enumerate(test_dataset.items()):
        test_map[k] = idx
        test_data.append(data)
    
    print(f"Generating features for train ({len(train_data)} samples), val ({len(val_data)} samples), test ({len(test_data)} samples)...")
    train_dataset = generate_all_features(train_data, model, vocab, batch_size=batch_size)
    val_dataset = generate_all_features(val_data, model, vocab, batch_size=batch_size)
    test_dataset = generate_all_features(test_data, model, vocab, batch_size=batch_size)

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "train_map": train_map,
        "val_map": val_map,
        "test_map": test_map,
    }

def load_few_shot_dataset(task_dataset, task_path, task_name):
    all_shot_data = json.load(open(os.path.join(task_path, "all_shots_data.json")))[task_name]

    all_shot_datasets = {}
    for shot, repeat_data in tqdm(all_shot_data.items()):
        shot_dataset = {}
        for repeat_idx, shot_info in repeat_data.items():
            train_k_indices = []
            val_k_indices = []

            for patient_id, prediction_time in zip(shot_info["patient_ids_train_k"], shot_info["label_times_train_k"]):
                train_k_indices.append(task_dataset["train_map"][(str(patient_id), prediction_time)])

            for patient_id, prediction_time in zip(shot_info["patient_ids_val_k"], shot_info["label_times_val_k"]):
                val_k_indices.append(task_dataset["val_map"][(str(patient_id), prediction_time)])

            train_k_features = task_dataset["train"]["features"][train_k_indices]
            train_k_labels = task_dataset["train"]["labels"][train_k_indices]
            val_k_features = task_dataset["val"]["features"][val_k_indices]
            val_k_labels = task_dataset["val"]["labels"][val_k_indices]

            shot_dataset[repeat_idx] = {
                "train_k": {"features": train_k_features, "labels": train_k_labels},
                "val_k": {"features": val_k_features, "labels": val_k_labels},
            }

        all_shot_datasets[shot] = shot_dataset
    
    all_shot_datasets["test"] = task_dataset["test"]

    return all_shot_datasets


def load_model_on_device(pretrained_model_path, device):
    """Load model on a specific device."""
    # Set environment variable for torch
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    
    from get_embedding import GPT, Config
    config = Config.from_name("pythia-160m")
    config.padded_vocab_size = 185138
    # config_data = json.load(open(os.path.join(pretrained_model_path, "lit_config.json")))
    # config_data = config

    # config = Config(**config_data)
    model = GPT(config).to(device)
    checkpoint = torch.load(os.path.join(pretrained_model_path, "lit_model.pth"), map_location=device) 
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def process_single_task(task, patients, benchmark_path, split_file, batch_size, model_size, pretrained_model_path, gpu_id=None, overwrite=False):
    """Process a single task for parallel execution."""
    task_path = os.path.join(benchmark_path, task)
    split_path = os.path.join(benchmark_path, split_file)
    output_path = os.path.join(task_path, f"task_dataset_{model_size}.pkl")
    all_shot_output_path = os.path.join(task_path, f"all_shot_datasets_{model_size}.pkl")

    # Check if both output files already exist (only if not overwriting)
    if not overwrite and os.path.exists(output_path) and os.path.exists(all_shot_output_path):
        print(f"[GPU {gpu_id}] Skipped {task} - both output files already exist")
        return f"Skipped {task} - both output files already exist"

    try:
        # Set GPU device if specified
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
        else:
            device = "cuda"
        
        print(f"[GPU {gpu_id}] Starting processing task: {task}")
        
        # Load model and vocab within the worker process on the specified device
        print(f"[GPU {gpu_id}] Loading model for task: {task}")
        model = load_model_on_device(pretrained_model_path, device)
        vocab = json.load(open("vocabulary.json"))
        print(f"[GPU {gpu_id}] Model loaded successfully for task: {task}")
        
        # Only process if task_dataset doesn't exist or if overwriting
        if not os.path.exists(output_path) or overwrite:
            print(f"[GPU {gpu_id}] Processing task_dataset for: {task}")
            task_dataset = load_labeled_dataset(patients, task_path, split_path, model, vocab, batch_size)
            pickle.dump(task_dataset, open(output_path, "wb"))
            print(f"[GPU {gpu_id}] task_dataset saved for: {task}")
        else:
            # Load existing task_dataset
            print(f"[GPU {gpu_id}] Loading existing task_dataset for: {task}")
            task_dataset = pickle.load(open(output_path, "rb"))
        
        # Only process all_shot_datasets if it doesn't exist or if overwriting
        if not os.path.exists(all_shot_output_path) or overwrite:
            print(f"[GPU {gpu_id}] Processing all_shot_datasets for: {task}")
            all_shot_datasets = load_few_shot_dataset(task_dataset, task_path, task)
            pickle.dump(all_shot_datasets, open(all_shot_output_path, "wb"))
            print(f"[GPU {gpu_id}] all_shot_datasets saved for: {task}")
        else:
            print(f"[GPU {gpu_id}] all_shot_datasets already exists for: {task}")
        
        print(f"[GPU {gpu_id}] Successfully completed task: {task}")
        return f"Completed {task}"
    except Exception as e:
        error_msg = f"Error processing {task}: {str(e)}"
        print(f"[GPU {gpu_id}] {error_msg}")
        return error_msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="160m", choices=["160m", "1b", "2.4b"],
                        help="Model size to use for generating features.")
    parser.add_argument("--overwrite", action="store_true", default=True,
                        help="Overwrite existing output files if they already exist.")
    args = parser.parse_args()

    split_file = "person_id_map.csv"
    benchmark_path = "benchmark"
    all_tasks = [
        #'chexpert',
        # 'new_lupus',
        # 'new_hypertension',
        # 'lab_thrombocytopenia',
        # 'guo_icu',
        # 'lab_hypoglycemia',
        # 'new_hyperlipidemia',
        # 'guo_los',
        # 'guo_readmission',
        # 'lab_hyponatremia',
        # 'lab_hyperkalemia',
        # 'new_celiac',
        'new_pancan',
        # 'lab_anemia',
        # 'new_acutemi'
    ]

    patients = load_patients("ehrshot_in_nhird_patients_v800.json")
    if args.model_size == "1b":
        pretrained_model_path = "CatchFM-1b"
        batch_size = 16
    elif args.model_size == "2.4b":
        pretrained_model_path = "CatchFM-2.4b"
        batch_size = 16
    else:
        pretrained_model_path = "/data/stevenz3/EHR/finetune/finetune_0.85"    
        batch_size = 16

    # Get number of available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s) available")
    
    # Use number of GPUs for parallel processing, but cap at number of tasks
    n_jobs = min(n_gpus, len(all_tasks))
    print(f"Using {n_jobs} parallel jobs")
    
    # Process tasks in parallel with GPU assignment and progress bar
    print(f"\n{'='*60}")
    print(f"Starting parallel processing of {len(all_tasks)} tasks using {n_jobs} GPUs")
    print(f"Each GPU will process {len(all_tasks) // n_jobs + (1 if len(all_tasks) % n_jobs else 0)} tasks")
    print(f"Using ThreadPoolExecutor for dynamic task assignment")
    if args.overwrite:
        print(f"Overwrite mode: Will overwrite existing output files")
    print(f"{'='*60}\n")
    
    # Create a progress bar for the overall task processing
    with tqdm(total=len(all_tasks), desc="Processing Tasks", unit="task") as pbar:
        # Use a thread pool to dynamically assign tasks as GPUs become available
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # Set environment variable for torch in the main process
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        
        # Create a thread-safe task queue
        task_queue = all_tasks.copy()
        results = [None] * len(all_tasks)  # Pre-allocate results list
        task_lock = threading.Lock()
        result_lock = threading.Lock()
        
        def worker(gpu_id):
            """Worker function that processes tasks for a specific GPU"""
            # Set environment variable in each worker thread
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            
            while True:
                # Get next task from queue
                with task_lock:
                    if not task_queue:
                        break
                    task = task_queue.pop(0)
                    task_index = len(all_tasks) - len(task_queue) - 1
                
                print(f"[GPU {gpu_id}] Starting task: {task}")
                
                # Process the task
                result = process_single_task(
                    task, patients, benchmark_path, split_file, batch_size, 
                    args.model_size, pretrained_model_path, gpu_id, args.overwrite
                )
                
                # Store result and update progress
                with result_lock:
                    results[task_index] = result
                    pbar.update(1)
                    pbar.set_postfix_str(f"GPU {gpu_id}: {result}")
                
                print(f"[GPU {gpu_id}] Completed task: {task}")
        
        # Start worker threads for each GPU
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Submit worker tasks for each GPU
            futures = [executor.submit(worker, i) for i in range(n_jobs)]
            
            # Wait for all workers to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker error: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Filter out None results (shouldn't happen, but just in case)
        results = [r for r in results if r is not None]
    
    # Print final summary
    print(f"\n{'='*60}")
    print("Processing Summary:")
    print(f"{'='*60}")
    completed = sum(1 for r in results if "Completed" in r)
    skipped = sum(1 for r in results if "Skipped" in r)
    errors = sum(1 for r in results if "Error" in r)
    
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Completed: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"{'='*60}")
    
    # Print detailed results
    print("\nDetailed Results:")
    for i, result in enumerate(results):
        print(f"Task {i+1:2d}: {result}")
