import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)



class Evaluator:
    def __init__(self, task="binary", threshold=0.5):
        """
        task: "binary"
        threshold: soglia decisionale per classificazione binaria
        """
        self.task = task
        self.threshold = threshold

    def evaluate(self, logits, labels, loss=None):
        """
        logits: tensor (N,1) o (N,)
        labels: tensor (N,) o (N,1)
        loss: opzionale
        """

        logits = logits.detach().cpu()
        labels = labels.detach().cpu()

        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.view(-1)

        if labels.ndim == 2:
            labels = labels.view(-1)

        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).long()

        y_true = labels.numpy()
        y_pred = preds.numpy()
        y_prob = probs.numpy()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = float("nan")

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "specificity": specificity,
            "f1": f1,
            "auc": auc,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        if loss is not None:
            metrics["loss"] = loss

        return metrics

    @staticmethod
    def print_report(metrics):

        print("\n========== METRICS ==========")
        for k in ["loss", "accuracy", "precision", "recall", "specificity", "f1", "auc"]:
            if k in metrics:
                print(f"{k:12s}: {metrics[k]:.4f}")

        print("\nConfusion Matrix")
        print("              Pred 0    Pred 1")
        print(f"Actual 0     {metrics['tn']:6d}     {metrics['fp']:6d}")
        print(f"Actual 1     {metrics['fn']:6d}     {metrics['tp']:6d}")
        print("================================\n")
    
    def evaluate_bootstrap(
        self,
        logits,
        labels,
        loss=None,
        n_bootstrap=1000,
        ci=95,
        seed=42
    ):
        """
        Bootstrap non parametrico sulle metriche.
        Restituisce media e intervalli di confidenza.
        """

        torch.manual_seed(seed)
        np.random.seed(seed)

        logits = logits.detach().cpu()
        labels = labels.detach().cpu()

        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.view(-1)

        if labels.ndim == 2:
            labels = labels.view(-1)

        probs = torch.sigmoid(logits).numpy()
        y_true = labels.numpy()

        N = len(y_true)

        # Contenitore per metriche bootstrap
        bootstrap_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "specificity": [],
            "f1": [],
            "auc": [],
        }

        for _ in range(n_bootstrap):

            indices = np.random.choice(N, size=N, replace=True)

            y_t = y_true[indices]
            y_p = probs[indices]
            y_pred = (y_p >= self.threshold).astype(int)

            acc = accuracy_score(y_t, y_pred)
            prec = precision_score(y_t, y_pred, zero_division=0)
            rec = recall_score(y_t, y_pred, zero_division=0)
            f1 = f1_score(y_t, y_pred, zero_division=0)

            try:
                auc = roc_auc_score(y_t, y_p)
            except:
                auc = np.nan

            tn, fp, fn, tp = confusion_matrix(y_t, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            bootstrap_metrics["accuracy"].append(acc)
            bootstrap_metrics["precision"].append(prec)
            bootstrap_metrics["recall"].append(rec)
            bootstrap_metrics["specificity"].append(specificity)
            bootstrap_metrics["f1"].append(f1)
            bootstrap_metrics["auc"].append(auc)

        results = {}

        alpha = 100 - ci
        lower_q = alpha / 2
        upper_q = 100 - alpha / 2

        for metric, values in bootstrap_metrics.items():
            values = np.array(values)

            results[metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "ci_lower": float(np.nanpercentile(values, lower_q)),
                "ci_upper": float(np.nanpercentile(values, upper_q)),
            }

        if loss is not None:
            results["loss"] = loss

        return results

    def plot_roc_curve(self, logits, labels, figsize=(6,6), save_path=None):
        """
        Plotta la ROC curve per classificazione binaria.

        logits: tensor (N,) o (N,1)
        labels: tensor (N,) o (N,1)
        figsize: dimensione del plot
        save_path: se specificato, salva il plot
        """

        logits = logits.detach().cpu()
        labels = labels.detach().cpu()

        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.view(-1)

        if labels.ndim == 2:
            labels = labels.view(-1)

        y_true = labels.numpy()
        y_prob = torch.sigmoid(logits).numpy()

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.savefig("plots/ROC.png",bbox_inches='tight')