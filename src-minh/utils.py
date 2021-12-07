# Performance metric functions

from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, log_loss
import numpy as np

def print_metrics(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    preds = preds.squeeze()

    # this is the binary cross-entropy loss, same as in training
    print("Loss:\t", log_loss(labels, preds))
    print("auROC:\t", roc_auc_score(labels, preds))
    auPRC = average_precision_score(labels, preds)
    print("auPRC:\t", auPRC)
    print_confusion_matrix(preds, labels)
    return auPRC

def print_confusion_matrix(preds, labels):
    npthresh = np.vectorize(lambda t: 1 if t >= 0.5 else 0)
    preds_binarized = npthresh(preds)
    conf_matrix = confusion_matrix(labels, preds_binarized)
    print("Confusion Matrix (at t = 0.5):\n", conf_matrix)