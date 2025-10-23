import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import get_cosine_schedule_with_warmup
from types import SimpleNamespace
from torch.optim import AdamW
from torch.utils.data import DataLoader,Subset
from tqdm.auto import tqdm, trange
from torch.utils.data.distributed import DistributedSampler
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
import json
import pickle
import csv 
import math
import re

from lit_gpt.model_code import GPT, Block, Config
from lit_gpt.utils import (
    check_nvlink_connectivity,
    get_default_supported_precision,
    lazy_load,
    num_parameters,
    eval_metrics,
    convert_to_numpy,
    load_checkpoint
)

import wandb
from IPython import embed


def setup(
    model_name: str = "pythia-70m",
    train_data_file: str = "/home/liwens/healthcare/Lightning-Pretrain/data/bp_patients/finetune/train.pkl",
    eval_data_file: str = "/home/liwens/healthcare/Lightning-Pretrain/data/bp_patients/finetune/valid.pkl",
    output_dir: str = "/data/group_data/cx_healthcare/nhird_finetune/pythia_70m_2048/empty/bp_patients",
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.1,
    decay_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_gradient_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    num_train_epochs: int = 2,
    dataloader_num_workers: int = 10,
    logging_steps: int = 10,
    num_eval_per_epoch: int = 6,
    max_steps: int = -1,
    precision: Optional[str] = None,
    devices: int = 1,
    is_test: bool = False,
    pretrained_model_path: Optional[str] = None,
    seed: int = 42
) -> None:
    args = locals()
    args = SimpleNamespace(**args)
    precision = precision or get_default_supported_precision(training=True)
    fabric_devices = devices
    if fabric_devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"
    

    fabric = L.Fabric(devices=fabric_devices, strategy=strategy, precision=precision)
    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)
    if args.is_test:
        fabric.launch(test,args)    
    else:
        fabric.launch(main,args)


def main(
    fabric: L.Fabric,
    args: Dict,
) -> None:

    fabric.seed_everything(args.seed)  # same seed for every process to init model (FSDP)
    if fabric.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        setup_wandb(args)    
        
    config = Config.from_name(args.model_name)
    config.padded_vocab_size = 185138

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
      
    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")        
    model = fabric.setup(model)
        
    train_dataloader = get_dataloader(args.train_data_file,  args.per_device_train_batch_size, args.dataloader_num_workers, config.block_size)
    eval_dataloader = get_dataloader(args.eval_data_file, args.per_device_eval_batch_size, args.dataloader_num_workers, config.block_size)
    train_dataloader,eval_dataloader = fabric.setup_dataloaders(train_dataloader, eval_dataloader)
    
    if args.max_steps > 0:
        t_total = args.max_steps
        num_train_epochs = (
            args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = int(len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)
        num_train_epochs = args.num_train_epochs

    args.eval_steps = t_total//(args.num_eval_per_epoch*num_train_epochs)

    
    no_decay = ["bias", "norm_1.weight", "norm_2.weight", "ln_f.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, foreach=False)
    optimizer = fabric.setup_optimizers(optimizer)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_ratio*t_total, num_training_steps=t_total
    # )
    
    scheduler = get_wsd_scheduler(optimizer, warmup_iters = int(args.warmup_ratio*t_total),
                                  stable_iters =int((1-args.decay_ratio)*t_total), max_iters = t_total)

    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "global_step": 0,
        "epoch":0
    }
    
    if args.pretrained_model_path is None:
        model.apply(model._init_weights)
        fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")        
    else:
        load_checkpoint(fabric, state["model"], args.pretrained_model_path)
        fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr) 
           
    total_train_batch_size = (
        args.per_device_train_batch_size*
        args.gradient_accumulation_steps*
        fabric.world_size 
    )
    fabric.print("***** Running training *****")
    fabric.print(f"  Num examples = {len(train_dataloader.dataset)}")
    fabric.print(f"  Num Epochs = {num_train_epochs}")
    fabric.print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    fabric.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    fabric.print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    fabric.print(f"  Total optimization steps = {t_total}")


    train_time = time.perf_counter()
    train(fabric, state, args, train_dataloader,eval_dataloader)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(
    fabric: L.Fabric,
    state: dict,
    args: dict,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]

    model.train()
    best_metric= -1    
    train_iterator = trange(
        state['epoch'], int(args.num_train_epochs), desc="Epoch", disable=not fabric.is_global_zero
    )   
    
    for epoch in train_iterator: 
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)   
        epoch_iterator = tqdm(train_dataloader, desc="Finetuning", disable=not fabric.is_global_zero)
        for step, inputs in enumerate(epoch_iterator):

            no_accumulating = (step + 1) % args.gradient_accumulation_steps == 0 
            
            with fabric.no_backward_sync(model, enabled=not no_accumulating):
                logits = model(inputs["input_ids"],inputs["positions"],classify=True)
                loss = torch.nn.functional.cross_entropy(logits, inputs["labels"])
                fabric.backward(loss / args.gradient_accumulation_steps)

            if no_accumulating:
                global_grad_norm = fabric.clip_gradients(model, optimizer, max_norm=args.max_gradient_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                state["global_step"] += 1
                state["epoch"] = epoch + (step + 1) / len(epoch_iterator)
                
                if (args.logging_steps > 0 and state["global_step"] % args.logging_steps == 0): 
                    loss_item = loss.item()  # expensive device-to-host synchronization
                    if fabric.global_rank == 0:
                        epoch_iterator.set_description(f'Loss: {loss_item}')
                        if not os.getenv("WANDB_DISABLED"):
                            wandb.log(
                                {
                                    "train/loss": loss_item,
                                    "learning_rate": scheduler.get_lr()[0],
                                    "train/global_grad_norm": global_grad_norm.item()
                                },
                                step = state["global_step"]
                            )
                            
                if  state["global_step"] % args.eval_steps == 0:
                    t0 = time.perf_counter()
                    results = validate(fabric, model, eval_dataloader,args)
                    t1 = time.perf_counter() - t0
                    fabric.print(f"step {state['global_step']}: val loss {results['valid/loss']:.4f}, val time: {t1 * 1000:.2f}ms")
                    results = fabric.all_reduce(results)
                    for key, value in results.items():
                        fabric.print(f"{key:25}: {value}")                    
                    if fabric.global_rank == 0:
                        if not os.getenv("WANDB_DISABLED"):
                            wandb.log(
                                results,
                                step = state["global_step"]
                            )
                    fabric.barrier()
                    if results["Sensitivity"] > best_metric and results["Specificity"] >= 0.985:
                        best_metric =  results["Sensitivity"]                                
                        output_path = os.path.join(args.output_dir, f"best-{args.seed}.pth")
                        fabric.print(f"Saving model checkpoint to {output_path}")
                        fabric.save(output_path, state)   
                                    
            if args.max_steps > 0 and step["global_step"] > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and step["global_step"] > args.max_steps:
            train_iterator.close()
            break



def test(fabric: L.Fabric, args: Dict)-> None:


    fabric.seed_everything(args.seed)
    
    config = Config.from_name(args.model_name)
    config.padded_vocab_size = 185138

        
    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

    t0 = time.perf_counter()
    load_checkpoint(fabric, model, os.path.join(args.output_dir, f"best-{args.seed}.pth"))
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)    
    model = fabric.setup(model)
    
    eval_dataloader = get_dataloader(args.eval_data_file, args.per_device_eval_batch_size, args.dataloader_num_workers, config.block_size)
    eval_dataloader = fabric.setup_dataloaders(eval_dataloader)
    
    results = validate(fabric, model, eval_dataloader,args,is_test = True)
    for key, value in results.items():
        fabric.print(f"{key:25}: {value}")    
    results.update({
        'Seed':args.seed
    })
    del results['valid/loss']
    
    # Extract the substring after 'test_' and before '.pkl'
    match = re.search(r'test_(.+)\.pkl', args.eval_data_file)
    if match:
        suffix = match.group(1)  # Extract the part after 'test_'
        # Define the output file name
        output_file = f"results_{suffix}.csv"
        print(f"Extracted suffix: {suffix}")
        print(f"Output file name: {output_file}")
        out_file = os.path.join(args.output_dir, output_file)
    else:
        print("No valid suffix found in the file path.")
        out_file = os.path.join(args.output_dir, "results.csv")
        
    file_exists = os.path.isfile(out_file)
    with open(out_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
        
        

@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, eval_dataloader: DataLoader, args: Dict, is_test: bool = False ) -> Dict:
    batch_size = eval_dataloader.batch_size
    eval_dataset_len =  len(eval_dataloader)
    if is_test:
        fabric.print("***** Running Testing Evaluation *****")
    else:
        fabric.print("***** Running Validation Evaluation *****")
    
    fabric.print(f"  Num batches = {eval_dataset_len}")
    fabric.print(f"  Batch size = {batch_size}")
    
    model.eval()
    total_preds = []
    total_targets = []
    eval_losses = torch.zeros(eval_dataset_len, device=fabric.device) 
    for k,inputs in enumerate(tqdm(eval_dataloader, desc=f"Rank {fabric.local_rank} Evaluating",disable=not fabric.is_global_zero)):  
        logits = model(inputs["input_ids"],inputs["positions"],classify=True)
        loss = torch.nn.functional.cross_entropy(logits, inputs["labels"])
        logits = torch.nn.functional.softmax(logits,dim=1)
        eval_losses[k] = loss.item()
        total_preds.append(logits[:,1])
        total_targets.append(inputs["labels"])
        
    val_loss = eval_losses.mean()
    total_preds = torch.cat(total_preds,dim=0)
    total_targets = torch.cat(total_targets,dim=0)

    total_preds,total_targets = convert_to_numpy(total_preds,total_targets)
    if is_test:
        with open(os.path.join(args.output_dir,f"preds_and_targets_{args.seed}_fujenall.pkl"), "wb") as f:
            pickle.dump({
                "preds": total_preds,
                "targets": total_targets
            }, f)        
    f1,roc_auc,accuracy,average_precision,sensitivity,specificity = eval_metrics(total_preds,total_targets)
    results = {
        "F1":f1,
        "Accuracy":accuracy,
        "AUC":roc_auc,
        "AUPRC":average_precision,
        "Sensitivity":sensitivity,
        "Specificity":specificity,
        "valid/loss":val_loss.item()
    }

    model.reset_cache()
    model.train()
    return results

def get_dataloader(data_file_path: str,batch_size:int, num_workers: int, block_size: int)->DataLoader:
    nhird_dataset = NHIRDDataset(data_file_path,block_size)
    nhird_dataloader =  DataLoader(
            nhird_dataset,
            batch_size=batch_size,
            collate_fn=nhird_dataset.collator,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True
        )
    return nhird_dataloader

class NHIRDDataset(Dataset):
    def __init__(self, data_file: Path, block_size: int):
        super().__init__()
        self.data_file = pickle.load(open(data_file,'rb'))
        self.block_size = block_size

    def __getitem__(self, idx):
        return self.data_file[idx]
    
    def __len__(self):
        return len(self.data_file)
    
    def collator(self,batch):
        input_ids = [torch.tensor(data["input_ids"]).type(torch.int64) for data in batch]
        positions = [torch.tensor(data["positions"]).type(torch.int64) for data in batch]
        labels = [torch.tensor(data["label"]).type(torch.int64) for data in batch]
        max_len = min(self.block_size, max(len(s) for s in input_ids))

        def pad_right(x, pad_id=0,pos=False):
            n = max_len - len(x)
            if n > 0:
                return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))
            else:
                x[max_len-1] = 2 if not pos else x[max_len-2]+1 
                return x[:max_len]  # Truncate to block size if sequence is too long

        # Pad or truncate and stack the input_ids, positions, and position_days
        padded_input_ids = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
        padded_positions = torch.stack([pad_right(x, pad_id=0,pos=True) for x in positions])

        return {
            'input_ids': padded_input_ids,
            'positions': padded_positions,
            'labels':  torch.stack(labels)
        }    


def get_wsd_scheduler(optimizer, warmup_iters: int, stable_iters: int, max_iters: int)-> torch.optim.lr_scheduler.LambdaLR:
    """
    Returns a learning rate scheduler based on the Warmup-Stable-Decay schedule.
    
    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        warmup_iters (int): Number of warmup iterations (W).
        stable_iters (int): Number of stable training iterations (T).
        max_iters (int): Total number of iterations (S).
        learning_rate (float): The maximum learning rate (η).
        
    Returns:
        LambdaLR: The learning rate scheduler.
    """
    def lr_lambda(it):
        if it < warmup_iters:
            # Warmup phase: linear increase from 0 to max_lr
            return it / warmup_iters
        elif it < stable_iters:
            # Stable phase: constant learning rate
            return 1.0
        else:
            # Decay phase: exponential annealing 
            return math.pow(0.5, (it - stable_iters) / (max_iters - stable_iters))
    # Apply the LambdaLR scheduler
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def setup_wandb(args: Dict):
    if not os.getenv("WANDB_DISABLED"):
        # wandb.init(project=os.getenv("WANDB_PROJECT", "healthcare"), entity=os.getenv("WANDB_ENTITY", "zhiyuan-chenyan-zhenghao-group"),
        #         group=os.getenv("WANDB_GROUP", "Foundation Finetuning"),name=os.getenv("WANDB_NAME", "Foundation Model Finetuning LitGPT"), config=args)    
        wandb.init(
            project="my-healthcare-project",       # 👈 new or existing project name under your account
            entity="qingyzhang-carnegie-mellon-university",  # 👈 your username (check exactly in W&B profile)
            group="Foundation Finetuning",
            name="Foundation Model Finetuning LitGPT",
            config=args
            )


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)