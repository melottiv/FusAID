

from src.datasets.fusion_dataset import FusionDataset
import torch

class ConcatFusionDataset(torch.utils.data.Dataset):

    def __init__(self,
                 input_df,
                 seq_path,
                 struct_path,
                 ids,
                 task,
                 transform_seq=None,
                 transform_struct=None):

        self.seq_dataset = FusionDataset(
            input_df, seq_path, ids, task, transform=transform_seq
        )

        self.struct_dataset = FusionDataset(
            input_df, struct_path, ids, task, transform=transform_struct
        )

    def __len__(self):
        return len(self.seq_dataset)

    def __getitem__(self, idx):
        x_seq, y = self.seq_dataset[idx]
        x_struct, _ = self.struct_dataset[idx]
        x = torch.cat([x_seq, x_struct], dim=0)
        return x, y