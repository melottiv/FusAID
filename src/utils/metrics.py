# src/utils/metrics.py

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def compute_metrics(labels: torch.Tensor, probs: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Calcola metriche di classificazione binaria, inclusa la confusion matrix.

    Args:
        labels (torch.Tensor): Tensor 1D o 2D con valori 0 o 1 (ground truth).
        probs (torch.Tensor): Tensor 1D o 2D con probabilità predette (dallo sigmoid).
        threshold (float): Soglia per trasformare probabilità in predizioni binarie.

    Returns:
        dict: dizionario con accuracy, precision, recall, f1, auc e confusion matrix (TP, TN, FP, FN)
    """
    # Trasforma probabilità in predizioni binarie
    preds = (probs >= threshold).float()

    # Converti in numpy
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    probs_np = probs.cpu().numpy()

    metrics = {}
    metrics["accuracy"] = accuracy_score(labels_np, preds_np)
    metrics["precision"] = precision_score(labels_np, preds_np, zero_division=0)
    metrics["recall"] = recall_score(labels_np, preds_np, zero_division=0)
    metrics["f1"] = f1_score(labels_np, preds_np, zero_division=0)

    # AUC
    if len(set(labels_np)) > 1:
        metrics["auc"] = roc_auc_score(labels_np, probs_np)
    else:
        metrics["auc"] = float("nan")

    # Confusion matrix: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(labels_np, preds_np, labels=[0,1]).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)

    return metrics
