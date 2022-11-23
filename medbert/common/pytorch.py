import torch

def batch_to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch

def batch_detach(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.detach()
    return batch