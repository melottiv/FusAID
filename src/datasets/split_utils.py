import networkx as nx
import numpy as np
from collections import defaultdict

def split_data(df, val_ratio=0.15, test_ratio=0.15,
               seed=42, max_component_frac=0.01, verbose=False):

    if df.empty:
        return [], [], []

    #return split_loose(df, val_ratio, test_ratio, seed, verbose=True)
    if (df["label"] == "wt").any():
        return split_data_wt(df, val_ratio, test_ratio+0.05,
                             seed, max_component_frac, verbose)
    else:
        return split_data_onco(df, val_ratio, test_ratio,
                               seed, max_component_frac*5, verbose)



def split_data_wt(df,
                  val_ratio=0.2,
                  test_ratio=0.2,
                  seed=41,
                  max_component_frac=0.01,
                  verbose=False,slack=0.05):


    df_full = df.copy()

    wildtypes = df_full[df_full["label"] == "wt"]
    df = df_full[df_full["label"] != "wt"]
    if df.empty:
        return [], [], []



    assert "gene_h" in df.columns and "gene_t" in df.columns

    rng = np.random.default_rng(seed)

    G = nx.Graph()
    edge_to_rows = {}

    for idx, row in df.iterrows():
        g1 = row["gene_h"]
        g2 = row["gene_t"]
        G.add_edge(g1, g2)
        edge = tuple(sorted((g1, g2)))
        edge_to_rows.setdefault(edge, []).append(idx)

    if G.number_of_nodes() == 0:
        return [], [], []

    max_nodes = max(2, int(max_component_frac * G.number_of_nodes()))

    removed_rows = set()
    final_components = []

    from collections import deque
    queue = deque(nx.connected_components(G))

    while queue:
        comp = set(queue.popleft())

        if len(comp) <= max_nodes:
            final_components.append(comp)
            continue

        subG = G.subgraph(comp).copy()

        if subG.number_of_nodes() <= 2:
            final_components.append(comp)
            continue

        part1, part2 = nx.algorithms.community.kernighan_lin_bisection(
            subG, seed=seed
        )

        part1 = set(part1)
        part2 = set(part2)

        for u, v in subG.edges():
            if (u in part1 and v in part2) or (u in part2 and v in part1):
                edge = tuple(sorted((u, v)))
                for row_idx in edge_to_rows.get(edge, []):
                    removed_rows.add(row_idx)

        queue.append(part1)
        queue.append(part2)

    components = final_components



    valid_indices = [idx for idx in df.index if idx not in removed_rows]
    n_total = len(valid_indices)
    n_slack= int(slack * n_total)
    n_train_target = int((1 - val_ratio - test_ratio) * n_total)
    n_val_target   = int(val_ratio * n_total)
    n_test_target  = n_total - n_train_target - n_val_target

    train_idx, val_idx, test_idx = [], [], []
    counts = {"train": 0, "val": 0, "test": 0}

    gene_to_rows = {}
    for idx, row in df.iterrows():
        if idx in removed_rows:
            continue
        g1 = row["gene_h"]
        g2 = row["gene_t"]

        gene_to_rows.setdefault(g1, []).append(idx)
        gene_to_rows.setdefault(g2, []).append(idx)

    for comp in components:
        comp_rows = set()
        for gene in comp:
            comp_rows.update(gene_to_rows.get(gene, []))

        comp_rows = list(comp_rows)
        comp_size = len(comp_rows)

        if counts["train"] + comp_size <= n_train_target + n_slack:
            train_idx.extend(comp_rows)
            counts["train"] += comp_size
        elif counts["val"] + comp_size <= n_val_target + n_slack:
            val_idx.extend(comp_rows)
            counts["val"] += comp_size
        else:
            test_idx.extend(comp_rows)
            counts["test"] += comp_size


    train_genes = set(df.loc[train_idx]["gene_h"]) | set(df.loc[train_idx]["gene_t"])
    val_genes   = set(df.loc[val_idx]["gene_h"])   | set(df.loc[val_idx]["gene_t"])
    test_genes  = set(df.loc[test_idx]["gene_h"])  | set(df.loc[test_idx]["gene_t"])

    wt_train, wt_val, wt_test = [], [], []
    wt_unassigned = []

    for wt_idx, row in wildtypes.iterrows():
        gene = row["gene"]

        if gene in train_genes:
            wt_train.append(wt_idx)
        elif gene in val_genes:
            wt_val.append(wt_idx)
        elif gene in test_genes:
            wt_test.append(wt_idx)
        else:
            wt_unassigned.append(wt_idx)

    wt_total = len(wildtypes)

    val_total_target   = int(round(wt_total * val_ratio))
    test_total_target  = int(round(wt_total * test_ratio))
    train_total_target = wt_total - val_total_target - test_total_target

    targets = {
        "train": train_total_target,
        "val": val_total_target,
        "test": test_total_target
    }

    def current_counts():
        return {
            "train": len(wt_train),
            "val": len(wt_val),
            "test": len(wt_test)
        }

    for wt_pos in wt_unassigned:

        counts_now = current_counts()

        deficits = {
            p: targets[p] - counts_now[p]
            for p in ["train", "val", "test"]
        }

        positive_deficits = {p: d for p, d in deficits.items() if d > 0}

        if positive_deficits:
            best_partition = max(positive_deficits, key=positive_deficits.get)
        else:
            best_partition = min(counts_now, key=counts_now.get)

        if best_partition == "train":
            wt_train.append(wt_pos)
        elif best_partition == "val":
            wt_val.append(wt_pos)
        else:
            wt_test.append(wt_pos)

    n_wt_train = len(wt_train)
    n_wt_val   = len(wt_val)
    n_wt_test  = len(wt_test)

    train_total=len(train_idx)+n_wt_train
    val_total=len(val_idx)+n_wt_val
    test_total=len(test_idx)+n_wt_test

    n=train_total+val_total+test_total


    if n_wt_train == 0 or n_wt_val == 0 or n_wt_test == 0:
        raise RuntimeError(
            f"Invalid WT split: "
            f"train={n_wt_train}, val={n_wt_val}, test={n_wt_test}. "
            "Each partition must contain at least one WT sample."
        )

    train_idx.extend(wt_train)
    val_idx.extend(wt_val)
    test_idx.extend(wt_test)

    return train_idx, val_idx, test_idx


def split_data_onco(df,
                    val_ratio=0.15,
                    test_ratio=0.15,
                    seed=41,
                    max_component_frac=0.02,
                    verbose=False):


    assert "gene_h" in df.columns and "gene_t" in df.columns

    rng = np.random.default_rng(seed)

    df = df.copy()
    df["cancer_bin"] = (df["cancer"] != "Non-Cancer").astype(int)

    G = nx.Graph()
    for idx, row in df.iterrows():
        g1, g2 = row["gene_h"], row["gene_t"]
        G.add_edge(g1, g2)

    max_nodes = max(2, int(max_component_frac * G.number_of_nodes()))
    final_components = []

    from collections import deque
    queue = deque(nx.connected_components(G))

    while queue:
        comp = set(queue.popleft())

        if len(comp) <= max_nodes:
            final_components.append(comp)
            continue

        subG = G.subgraph(comp).copy()
        if subG.number_of_nodes() <= 2:
            final_components.append(comp)
            continue

        part1, part2 = nx.algorithms.community.kernighan_lin_bisection(subG, seed=seed)
        queue.append(set(part1))
        queue.append(set(part2))

    components = list(final_components)
    rng.shuffle(components)

    n_total = len(df)
    n_train_target = int((1 - val_ratio - test_ratio) * n_total)
    n_val_target   = int(val_ratio * n_total)
    n_test_target  = n_total - n_train_target - n_val_target
    global_ratio = df["cancer_bin"].mean()

    stats = {"train": set(), "val": set(), "test": set()}

    gene_to_rows = {}
    for idx, row in df.iterrows():
        g1, g2 = row["gene_h"], row["gene_t"]
        gene_to_rows.setdefault(g1, []).append(idx)
        gene_to_rows.setdefault(g2, []).append(idx)

    for comp in components:
        comp_rows = set()
        for gene in comp:
            comp_rows.update(gene_to_rows.get(gene, []))
        comp_rows = comp_rows - (stats["train"] | stats["val"] | stats["test"])  # rimuove duplicati
        if not comp_rows:
            continue

        comp_size = len(comp_rows)
        comp_cancer = df.loc[list(comp_rows)]["cancer_bin"].sum()

        best_split = None
        best_cost = float("inf")

        for split in ["train", "val", "test"]:
            current_n = len(stats[split])
            target_n = {"train": n_train_target, "val": n_val_target, "test": n_test_target}[split]

            if current_n + comp_size > target_n:
                continue

            new_ratio = (df.loc[list(stats[split])]["cancer_bin"].sum() + comp_cancer) / (current_n + comp_size)
            strat_error = abs(new_ratio - global_ratio)
            size_error = abs((current_n + comp_size) - target_n) / n_total
            cost = strat_error + size_error

            if cost < best_cost:
                best_cost = cost
                best_split = split

        if best_split is None:
            best_split = min(stats, key=lambda s: len(stats[s]))

        stats[best_split].update(comp_rows)

    train_idx = list(stats["train"])
    val_idx   = list(stats["val"])
    test_idx  = list(stats["test"])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    all_idx = set(train_idx + val_idx + test_idx)
    assert len(all_idx) == len(train_idx) + len(val_idx) + len(test_idx)
    assert all_idx.issubset(df.index)

    def print_split_stats(name, idx_list):
        n_split = len(idx_list)
        ratio_split = n_split / n_total

        if n_split == 0:
            print(f"{name}: EMPTY")
            return

        split_df = df.loc[idx_list]
        onco_ratio = split_df["cancer_bin"].mean()
        non_onco_ratio = 1.0 - onco_ratio

        print(f"\n{name.upper()} SET")
        print(f"  Samples: {n_split} ({ratio_split:.3f} of total)")
        print(f"  ONCO:     {onco_ratio:.3f}")
        print(f"  NON-ONCO: {non_onco_ratio:.3f}")


    return train_idx, val_idx, test_idx

def split_loose(df,
                val_ratio=0.15,
                test_ratio=0.15,
                seed=41,
                verbose=True):

    if df.empty:
        return [], [], []

    rng = np.random.default_rng(seed)


    if "label" in df.columns and (df["label"] == "wt").any():

        if verbose:
            print("Loose split stratified on label (wt vs fusion)")

        strat_col = df["label"].apply(
            lambda x: "wt" if x == "wt" else "fusion"
        )

    else:

        if "cancer" not in df.columns:
            raise ValueError("No 'cancer' column found for stratification")

        if verbose:
            print("Loose split stratified on cancer (Non-Cancer vs Cancer)")

        strat_col = df["cancer"].apply(
            lambda x: "Non-Cancer" if x == "Non-Cancer" else "Cancer"
        )


    train_idx, val_idx, test_idx = [], [], []

    for cls in strat_col.unique():

        cls_indices = strat_col[strat_col == cls].index.to_numpy()
        rng.shuffle(cls_indices)

        n_cls = len(cls_indices)

        n_train = int((1 - val_ratio - test_ratio) * n_cls)
        n_val   = int(val_ratio * n_cls)
        n_test  = n_cls - n_train - n_val

        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train:n_train+n_val])
        test_idx.extend(cls_indices[n_train+n_val:])

    if verbose:
        print("Split sizes:")
        print(f"Train: {len(train_idx)}")
        print(f"Val:   {len(val_idx)}")
        print(f"Test:  {len(test_idx)}")

        def print_dist(name, idx):
            subset = strat_col.loc[idx]
            dist = subset.value_counts(normalize=True)
            print(f"{name} distribution:")
            print(dist)

        print_dist("Train", train_idx)
        print_dist("Val", val_idx)
        print_dist("Test", test_idx)

    return train_idx, val_idx, test_idx


def gene_disjoint_kfold_stratified(
    df, 
    k=5, 
    seed=42, 
    max_component_size=None, 
    verbose=False
):
    """
    Gene-disjoint k-fold con stratificazione per 'cancer'.
    max_component_size: se un componente supera questa dimensione, viene splittato.
    """
    if df.empty:
        return []

    rng = np.random.default_rng(seed)

    # -----------------------------------------
    # Identifica se serve stratificazione
    # -----------------------------------------
    has_wt = (df["label"] == "wt").any()
    stratify = not has_wt

    # -----------------------------------------
    # Consideriamo solo fusioni per i componenti
    # -----------------------------------------
    fusion_df = df[df["label"] != "wt"].copy()
    fusion_df["cancer_bin"] = fusion_df["cancer"].apply(lambda x: 0 if x=="Non-Cancer" else 1)
    max_component_size=int(len(fusion_df)/(3*k))
    # Funzione per split grandi componenti
    def split_large_component(rows, max_size):
        if max_size is None or len(rows) <= max_size:
            return [rows]
        rows = list(rows)
        rng.shuffle(rows)
        return [rows[i:i+max_size] for i in range(0, len(rows), max_size)]

    # =====================================
    # Creazione componenti basati sul grafo completo
    # =====================================
    G = nx.Graph()
    for _, row in fusion_df.iterrows():
        G.add_edge(row["gene_h"], row["gene_t"])
    comp_list = list(nx.connected_components(G))
    rng.shuffle(comp_list)

    # Mappa gene -> righe
    gene_to_rows = defaultdict(list)
    for idx, row in fusion_df.iterrows():
        gene_to_rows[row["gene_h"]].append(idx)
        gene_to_rows[row["gene_t"]].append(idx)

    # Genera componenti come liste di indici
    components = []
    for comp in comp_list:
        rows = set()
        for gene in comp:
            rows.update(gene_to_rows.get(gene, []))
        # split se troppo grande
        for chunk in split_large_component(rows, max_component_size):
            components.append(list(chunk))

    # -----------------------------------------
    # Assegnamento greedy multi-criterio
    # -----------------------------------------
    folds = [set() for _ in range(k)]
    fold_sizes = [0] * k
    fold_counts = [defaultdict(int) for _ in range(k)]  # counts per classe
    fold_genes = [set() for _ in range(k)]  # geni presenti in ciascun fold

    for rows in components:
        comp_classes = fusion_df.loc[rows, "cancer_bin"].values
        comp_class_count = np.bincount(comp_classes, minlength=2)

        candidate_folds = []
        comp_genes = set(fusion_df.loc[rows, ["gene_h","gene_t"]].values.flatten())
        for i in range(k):
            if len(fold_genes[i] & comp_genes) == 0:
                candidate_folds.append(i)

        if not candidate_folds:
            overlaps = [len(fold_genes[i] & comp_genes) for i in range(k)]
            candidate_folds = [np.argmin(overlaps)]

        dominant_class = np.argmax(comp_class_count)
        best_fold = min(
            candidate_folds,
            key=lambda i: (fold_counts[i][dominant_class], fold_sizes[i])
        )

        folds[best_fold].update(rows)
        fold_sizes[best_fold] += len(rows)
        fold_genes[best_fold].update(comp_genes)
        for c in comp_classes:
            fold_counts[best_fold][c] += 1

    if has_wt:
        wt_df = df[df["label"] == "wt"]
        for idx, row in wt_df.iterrows():
            gene = row["gene"]
            assigned = False
            for i in range(k):
                if gene in fold_genes[i]:
                    folds[i].add(idx)
                    assigned = True
                    break
            if not assigned:
                smallest_fold = np.argmin([len(f) for f in folds])
                folds[smallest_fold].add(idx)


    folds = [list(f) for f in folds]

    if verbose:
        print("========== FOLD STATISTICS ==========")
        for i, f in enumerate(folds):
            print(f"Fold {i}: {len(f)} samples")
        if stratify:
            print("\n========== FOLD CANCER STATISTICS ==========")
            for i, f in enumerate(folds):
                cancer_vals = fusion_df.loc[f, "cancer_bin"]
                counts = cancer_vals.value_counts().to_dict()
                proportions = (cancer_vals.value_counts(normalize=True)).to_dict()
                print(f"Fold {i}")
                print(f"  Samples: {len(f)}")
                print(f"  Cancer counts (0=Non-Cancer, 1=Cancer): {counts}")
                print(f"  Cancer proportions: {proportions}")

    return folds

def gene_disjoint_kfold(
    df, 
    k=5, 
    seed=42, 
    max_component_size=None, 
    verbose=False
):

    if df.empty:
        return []

    rng = np.random.default_rng(seed)

    has_wt = (df["label"] == "wt").any()
    stratify = not has_wt

    fusion_df = df[df["label"] != "wt"]
    max_component_size=int(len(fusion_df)/(3*k))

    def split_large_component(rows, max_size):
        """Divide ricorsivamente se rows > max_size"""
        if max_size is None or len(rows) <= max_size:
            return [rows]
        
        rows = list(rows)
        rng.shuffle(rows)
        chunks = [rows[i:i+max_size] for i in range(0, len(rows), max_size)]
        return chunks

    components = []
    if stratify:
        fusion_df = fusion_df.copy()
        fusion_df["cancer_bin"] = fusion_df["cancer"].apply(lambda x: 0 if x=="Non-Cancer" else 1)
        for cls in [0,1]:
            cls_df = fusion_df[fusion_df["cancer_bin"] == cls]
            G = nx.Graph()
            for _, row in cls_df.iterrows():
                G.add_edge(row["gene_h"], row["gene_t"])
            cls_components = list(nx.connected_components(G))
            rng.shuffle(cls_components)

            gene_to_rows = defaultdict(list)
            for idx, row in cls_df.iterrows():
                g1, g2 = row["gene_h"], row["gene_t"]
                gene_to_rows[g1].append(idx)
                gene_to_rows[g2].append(idx)

            for comp in cls_components:
                rows = set()
                for gene in comp:
                    rows.update(gene_to_rows.get(gene, []))
             
                for chunk in split_large_component(rows, max_component_size):
                    components.append(list(chunk))
    else:
        G = nx.Graph()
        for _, row in fusion_df.iterrows():
            G.add_edge(row["gene_h"], row["gene_t"])
        comp_list = list(nx.connected_components(G))
        rng.shuffle(comp_list)

        gene_to_rows = defaultdict(list)
        for idx, row in fusion_df.iterrows():
            g1, g2 = row["gene_h"], row["gene_t"]
            gene_to_rows[g1].append(idx)
            gene_to_rows[g2].append(idx)

        for comp in comp_list:
            rows = set()
            for gene in comp:
                rows.update(gene_to_rows.get(gene, []))
            for chunk in split_large_component(rows, max_component_size):
                components.append(list(chunk))

    folds = [set() for _ in range(k)]
    fold_sizes = [0] * k

    for rows in components:
        smallest_fold = np.argmin(fold_sizes)
        folds[smallest_fold].update(rows)
        fold_sizes[smallest_fold] += len(rows)

    if has_wt:
        wt_df = df[df["label"] == "wt"]
        fold_genes = []
        for fold in folds:
            genes = set(df.loc[list(fold)]["gene_h"]) | set(df.loc[list(fold)]["gene_t"])
            fold_genes.append(genes)

        for idx, row in wt_df.iterrows():
            gene = row["gene"]
            assigned = False
            for i in range(k):
                if gene in fold_genes[i]:
                    folds[i].add(idx)
                    assigned = True
                    break
            if not assigned:
                smallest_fold = np.argmin([len(f) for f in folds])
                folds[smallest_fold].add(idx)

    folds = [list(f) for f in folds]

    if verbose:
        print("========== FOLD STATISTICS ==========")
        for i, f in enumerate(folds):
            print(f"Fold {i}: {len(f)} samples")
        if stratify:
            print("\n========== FOLD CANCER STATISTICS ==========")
            for i, f in enumerate(folds):
                cancer_vals = fusion_df.loc[f, "cancer"].apply(lambda x: 0 if x=="Non-Cancer" else 1)
                counts = cancer_vals.value_counts().to_dict()
                proportions = (cancer_vals.value_counts(normalize=True)).to_dict()
                print(f"Fold {i}")
                print(f"  Samples: {len(f)}")
                print(f"  Cancer counts (0=Non-Cancer, 1=Cancer): {counts}")
                print(f"  Cancer proportions: {proportions}")

    return folds
