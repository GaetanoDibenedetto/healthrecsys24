from sklearn.metrics import classification_report
import torch

def print_report(truelabels, predictions):

    if isinstance(truelabels, torch.Tensor):
        truelabels = truelabels.detach().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()
    
    print(classification_report(truelabels, predictions, digits=4))
