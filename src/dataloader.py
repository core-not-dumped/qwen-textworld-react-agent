import torch
from torch.utils.data import Dataset
from itertools import islice

class TeacherForcingDataloader:
    def __init__(self, dataset, batch_size=2, pad_token_id=0, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            torch.random.manual_seed(torch.randint(0, 10000, (1,)).item())
            self.indices = torch.randperm(len(self.dataset)).tolist()

        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
        batch_samples = [self.dataset[i] for i in batch_indices]
        self.current_idx += self.batch_size

        # padding 처리
        input_ids = [torch.tensor(sample["input_ids"]) for sample in batch_samples]
        labels = [torch.tensor(sample["labels"]) for sample in batch_samples]

        max_len = max(len(x) for x in input_ids)

        input_ids_padded = torch.stack([
            torch.cat([x, torch.full((max_len - len(x),), self.pad_token_id)]) for x in input_ids
        ])
        labels_padded = torch.stack([
            torch.cat([x, torch.full((max_len - len(x),), -100)]) for x in labels
        ])
        attention_mask = (input_ids_padded != self.pad_token_id).long()
        output_data = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask
        }

        return output_data

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size