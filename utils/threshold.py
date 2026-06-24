from sklearn.metrics import f1_score
import numpy as np
import torch

def search_alpha_threshold_f1(logits_seq, logits_struct, labels, n_a=21, n_t=101):

    logits_seq = logits_seq.view(-1)
    logits_struct = logits_struct.view(-1)
    labels = labels.view(-1).cpu().numpy()

    best = {
        "f1": -1,
        "alpha": 0.5,
        "threshold": 0.5
    }

    for a in np.linspace(0, 1, n_a):

        logits = a * logits_seq + (1 - a) * logits_struct
        probs = torch.sigmoid(logits).cpu().numpy()

        for t in np.linspace(0.01, 0.99, n_t):

            preds = (probs >= t).astype(int)
            f1 = f1_score(labels, preds)

            if f1 > best["f1"]:
                best = {
                    "f1": f1,
                    "alpha": a,
                    "threshold": t
                }

    return best



def estimate_best_a(logits_seq, logits_struct, labels, steps=101):

    from sklearn.metrics import roc_auc_score

    logits_seq = logits_seq.view(-1)
    logits_struct = logits_struct.view(-1)
    labels = labels.view(-1).cpu().numpy()

    best_a = 0.5
    best_score = -np.inf

    for a in np.linspace(0, 1, steps):

        logits = a * logits_seq + (1 - a) * logits_struct
        probs = torch.sigmoid(logits).cpu().numpy()


        score = roc_auc_score(labels, probs)

        if score > best_score:
            best_score = score
            best_a = a

    return best_a