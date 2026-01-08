import numpy as np
import torch


def get_batch(dataset: np.ndarray, batch_size, context_length, device="cpu"):
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[i : i + context_length]) for i in starts])
    y = torch.stack([torch.from_numpy(dataset[i + 1 : i + context_length + 1]) for i in starts])
    return x.to(device), y.to(device)
