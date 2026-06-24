import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class FusionDataset(Dataset):
    def __init__(self, metadata_df, emb_path, selected_ids=None, task=None,transform=None):
        self.transform = transform
        if not isinstance(metadata_df, pd.DataFrame):
            raise ValueError("FusionDataset ora richiede un DataFrame come metadata")

        if task not in ["is_fusion", "is_onco"]:
            raise ValueError("Task deve essere 'is_fusion' o 'is_onco'")

        self.mode = "from_files"

        # -------------------------
        # Load embeddings
        # -------------------------
        npz = np.load(emb_path)
        embeddings = npz["embeddings"].astype(np.float32)
        emb_indices = npz["indices"]

        index_to_row = {idx: i for i, idx in enumerate(emb_indices)}

        self.samples = []

        # -------------------------
        # Iterate CORRETTAMENTE sulle righe
        # -------------------------
        for ref_index, row in metadata_df.iterrows():

            if selected_ids is not None and ref_index not in selected_ids:
                continue

            if ref_index not in index_to_row:
                continue  

            # ----- label logic -----
            if task == "is_fusion":
                label = 1 if row["label"] == "fusion" else 0

            elif task == "is_onco":
                label = 0 if row["cancer"] == "Non-Cancer" else 1

            feat = embeddings[index_to_row[ref_index]]

            self.samples.append((ref_index, feat, label))

        if len(self.samples) == 0:
            raise ValueError("Dataset vuoto dopo filtro selected_ids.")

        self.input_dim = embeddings.shape[1]

        labels = [y for _,_, y in self.samples]
        print("Class distribution:", np.bincount(labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ref_index, feat, label = self.samples[idx]

        x = torch.tensor(feat, dtype=torch.float32)
        if self.transform:
            x = self.transform(x)

        y = torch.tensor(label, dtype=torch.float32)

        return x, y, ref_index