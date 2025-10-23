import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import torch
from lit_gpt.model_code import GPT, Block, Config
import json
import pickle
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader,Subset
from pretokenize_code import history_to_ids_sequence


class NHIRDDataset(Dataset):
    def __init__(self, dataset: Path, block_size: int, vocab: dict):
        super().__init__()
        self.dataset = self.preprocess_data(dataset, vocab)
        self.block_size = block_size

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
    
    def preprocess_data(self, dataset, vocab):
        processed_dataset = []
        for data in dataset:
            processed_dataset.append(self.tokenize_data(data, vocab))
        return processed_dataset

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
            "input_ids": padded_input_ids,
            "positions": padded_positions,
            "labels":  torch.stack(labels)
        }    
    
    def tokenize_data(self, data, vocab):
        input_ids, positions = history_to_ids_sequence(data["history"], vocab)
        return {
            "input_ids": input_ids,
            "positions": positions,
            "label": data["label"]
        }

def load_model(pretrained_model_path):
    config_data = json.load(open(os.path.join(pretrained_model_path, "lit_config.json")))
    config = Config(**config_data)
    model = GPT(config).to("cuda")
    checkpoint = torch.load(os.path.join(pretrained_model_path, "lit_model.pth"), map_location="cuda") 
    model.load_state_dict(checkpoint)
    model.eval()

    return model

def get_dataloader(dataset: list, batch_size:int, block_size: int, vocab: dict) -> DataLoader:
    nhird_dataset = NHIRDDataset(dataset, block_size, vocab)
    nhird_dataloader =  DataLoader(
            nhird_dataset,
            batch_size=batch_size,
            collate_fn=nhird_dataset.collator,
            num_workers=1,
            shuffle=False,
            pin_memory=False
        )
    return nhird_dataloader

def generate_all_features(dataset, model, vocab, batch_size=16, block_size=2048):
    dataloader = get_dataloader(dataset, batch_size=batch_size, block_size=block_size, vocab=vocab)

    all_features = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Generating features", unit="batch"):
        input_ids = batch["input_ids"].to("cuda")
        # print("Input IDs:", input_ids)
        positions = batch["positions"].to("cuda")
        # print("Positions:", positions)
        labels = batch["labels"]
        # print("Labels batch:", labels)
        outputs = model(input_ids, positions, classify=False, last_hidden=True)
        all_features.append(outputs.cpu().detach().numpy())
        # print("outputs:", outputs)
        all_labels += labels.cpu().detach().tolist()

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)

    return {
        "features": all_features,
        "labels": all_labels
    }


if __name__ == "__main__":
    pretrained_model_path_2b = "/data/group_data/cx_healthcare/nhird_pretrained/flop_2e21/pythia-2.4b/step-428192-ckpt-converted"
    pretrained_model_path_160m = "/data/group_data/cx_healthcare/nhird_pretrained/flop_1e18/pythia-160m/step-4800-ckpt-converted"    
    
    # Change 160m to 2.4b by replaceing pretrained_model_path_160m to pretrained_model_path_2.4b
    model = load_model(pretrained_model_path_160m)
    vocab = json.load(open("vocabulary.json"))
    dataset = json.load(open("/data/group_data/cx_healthcare/EHRSHOT/EHRSHOT-Benchmark-nhird-format/benchmark/guo_icu/all_shot_datasets.json"))["-1"]["0"]["train_k"]
    print("Creating train features")
    train_features = generate_all_features(model, dataset, vocab)
    pickle.dump(train_features, open("/data/group_data/cx_healthcare/EHRSHOT/EHRSHOT-Benchmark-nhird-format/benchmark/guo_icu/all_shot_train.pkl", "wb"))
    

