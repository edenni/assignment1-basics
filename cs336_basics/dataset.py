import numpy as np
import torch


def get_batch(dataset: np.ndarray, batch_size, context_length, device="cpu"):
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[i : i + context_length]) for i in starts])
    y = torch.stack([torch.from_numpy(dataset[i + 1 : i + context_length + 1]) for i in starts])
    return x.to(device), y.to(device)


class Dataset:
    def __init__(self, dataset_name: str, context_length: int, batch_size: int, device: str, **kwargs):
        dataset_path = f"data/{dataset_name}"
        self.train_data = np.memmap(f"{dataset_path}/train.bin", dtype=np.uint16, mode="r").astype(np.int64)
        self.val_data = np.memmap(f"{dataset_path}/val.bin", dtype=np.uint16, mode="r").astype(np.int64)
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        return get_batch(data, self.batch_size, self.context_length, self.device)


if __name__ == "__main__":
    dataset = Dataset("tinystories", 10, 2, "cuda:0")
    inputs, targets = dataset.get_batch("train")
    print(inputs, targets)
