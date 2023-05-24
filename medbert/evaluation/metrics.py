from typing import Any
import torch
from sklearn.metrics import accuracy_score


class PrecisionAtK:
    def __init__(self, topk=10):
        """Computes the precision@k for the specified value of k"""
        self.topk = topk

    def __call__(self, outputs, batch):
        logits = outputs.prediction_logits
        target = batch['target']
        
        ind = torch.where((target != -100) & (target != 0))

        logits = logits[ind]
        target = target[ind]

        _, pred = logits.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target)
        if correct.numel() == 0:
            return 0
        else:
            return correct.any(0).float().mean().item()
        

def binary_hit(outputs, batch, threshold=0.5):
    logits = outputs.prediction_logits
    target = batch['target']

    probs = torch.nn.functional.sigmoid(logits)
    predictions = (probs > threshold).long().view(-1)         # TODO: Add uncertainty measure

    correct = (predictions == target).float().mean().item()

    return correct
class Accuracy():
    def __init__(self) -> None:
        pass
    def __call__(self, outputs, batch) -> Any:
        logits = outputs.prediction_logits
        probas = torch.nn.functional.softmax(logits, dim=-1)
        _, predictions = torch.max(probas, dim=-1)
        return accuracy_score(predictions, batch['target'])

