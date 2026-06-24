import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as skPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA

def plot_labels(task, y):
    """Ritorna label testuali per le classi in base al task"""
    if task == "is_fusion":
        labels_map = {0: "wildtype", 1: "fusion"}
    elif task == "is_onco":
        labels_map = {0: "non onco", 1: "onco"}
    else:
        raise ValueError(f"Task {task} non supportato")
    return np.array([labels_map[int(v)] for v in y])

# -----------------------------
# PCA 2D
# -----------------------------
def PCA_plot(dataset,task):
    """
    dataset: FusionDataset
    Produce scatter plot 2D con colori/label delle classi
    """
    X = []
    y = []
    for i in range(len(dataset)):
        xi, yi = dataset[i]
        X.append(xi.numpy() if torch.is_tensor(xi) else xi)
        y.append(yi.numpy() if torch.is_tensor(yi) else yi)
    X = np.array(X)
    y = np.array(y).flatten()

    pca = skPCA(n_components=2)
    X_pca = pca.fit_transform(X)

    class_labels = plot_labels(task, y)

    plt.figure(figsize=(6,6))
    for lbl in np.unique(class_labels):
        idx = class_labels == lbl
        plt.scatter(X_pca[idx,0], X_pca[idx,1], label=lbl, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA 2D scatter - {task}")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/PCA.png")
    plt.close()

# -----------------------------
# LDA
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def LDA_plot(dataset, task):
    """
    dataset: FusionDataset
    Produce istogramma con distribuzioni della proiezione LDA
    Esegue anche una logistic regression veloce sulla componente LDA
    """
    X = []
    y = []
    for i in range(len(dataset)):
        xi, yi = dataset[i]
        X.append(xi.numpy() if torch.is_tensor(xi) else xi)
        y.append(yi.numpy() if torch.is_tensor(yi) else yi)
    X = np.array(X)
    y = np.array(y).flatten()

    # --- LDA 1D ---
    lda = skLDA(n_components=1)
    X_lda = lda.fit_transform(X, y).flatten()

    # --- Logistic Regression ---
    lr = LogisticRegression()
    X_lda_reshaped = X_lda.reshape(-1, 1)  # scikit-learn vuole shape (n_samples, n_features)
    lr.fit(X_lda_reshaped, y)
    y_pred = lr.predict(X_lda_reshaped)
    y_prob = lr.predict_proba(X_lda_reshaped)[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    print(f"[INFO] Logistic regression on LDA: Accuracy={acc:.3f}, AUC={auc:.3f}")

    # --- Plot histogram ---
    class_labels = plot_labels(task, y)
    plt.figure(figsize=(6,4))
    for lbl in np.unique(class_labels):
        plt.hist(X_lda[class_labels == lbl], bins=20, alpha=0.6, label=lbl)
    plt.xlabel("LDA component 1")
    plt.ylabel("Count")
    plt.title(f"LDA projection - {task}")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/LDA.png")