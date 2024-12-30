from torchevent.utils import spike2data
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch

def acc_metric_hook(outputs, targets):
    targets = torch.cat(targets)

    prediction = spike2data(torch.cat(outputs), return_pred=True)
    acc = (prediction == targets).sum().item()/len(prediction)
    
    return {"acc": acc}

def extented_cls_metric_hook(outputs, targets):
    prediction = spike2data(torch.cat(outputs), return_pred=True)
    
    prediction = prediction
    targets = torch.cat(targets)

    acc = (prediction == targets).sum().item()/len(prediction)
    precision = precision_score(targets, prediction, average='weighted', zero_division=0)
    recall = recall_score(targets, prediction, average='weighted', zero_division=0)
    f1 = f1_score(targets, prediction, average='weighted', zero_division=0)
    
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }