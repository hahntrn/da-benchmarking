# Performance metric functions

from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, log_loss
import numpy as np
import torch

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


def torch_cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov
    src: https://github.com/pytorch/pytorch/issues/19037
    """
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()
