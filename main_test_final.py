import argparse
import torch
import pandas as pd
import numpy as np
import os
import pickle

from src.models.single_model_pipeline import SingleModelPipeline
from src.datasets.concat_fusion_dataset import ConcatFusionDataset
from torch.utils.data import DataLoader
from src.datasets.split_utils import split_data
from src.config import Config
from src.datasets.fusion_dataset import FusionDataset
from src.models import build_model
from src.training.trainer import Trainer


# ==========================================================
# ARGS
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True,
                        choices=["sequence", "structure", "concat", "soft_voting"])
    parser.add_argument("--task", type=str, default=None)

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--checkpoint_seq", type=str)
    parser.add_argument("--checkpoint_struct", type=str)

    parser.add_argument("--reduced_db", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=str, default="predictions.tsv")
    parser.add_argument("--df", type=str, required=True)
    parser.add_argument("--seq_embs", type=str)
    parser.add_argument("--chem_embs", type=str)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    return parser.parse_args()


# ==========================================================
# UTILS
# ==========================================================

def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_params(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ==========================================================
# MAIN
# ==========================================================

def main():

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold=0.5 
    # ------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------

    input_df = pd.read_pickle(args.df)

    if args.mode in ["structure", "concat", "soft_voting"]:
        args.task = "is_onco"

    mask = np.ones(len(input_df), dtype=bool)

    if args.task == "is_onco":
        mask &= input_df["label"] == "fusion"

    if args.mode in ["structure", "concat", "soft_voting"] or args.reduced_db:
        emb_indices = np.load(args.chem_embs)["indices"]
        mask &= np.isin(input_df.index.values, emb_indices)

    input_df = input_df.loc[mask]

    # ------------------------------------------------------
    # SPLIT
    # ------------------------------------------------------

    _, _, test_ids = split_data(input_df,seed=args.seed,test_ratio=args.test_ratio,val_ratio=args.val_ratio)

    seq_path = args.seq_embs
    struct_path = args.chem_embs
    # ======================================================
    # MODE: SEQUENCE / STRUCTURE
    # ======================================================

    if args.mode in ["sequence", "structure"]:

        emb_path = seq_path if args.mode == "sequence" else struct_path
        scaler = load_scaler(args.checkpoint.replace(".pt", "_scaler.pkl"))

        def normalize(x):
            return torch.from_numpy(
                scaler.transform(x.numpy().reshape(1, -1)).squeeze(0)
            ).float()

        dataset = FusionDataset(input_df, emb_path, test_ids, args.task, transform=normalize)
        
        config = Config(mode=args.mode, train=False, test=True, checkpoint=args.checkpoint)
        loader   = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)


        model = build_model(config, input_dim=dataset[0][0].shape[0]).to(device)
        trainer = Trainer(model, torch.optim.Adam(model.parameters()), config, y_train=None)
        trainer.load_checkpoint(args.checkpoint)

        _, logits, labels,ids = trainer.evaluate(loader)
        params = load_params(
            args.checkpoint.replace(".pt", "_params.pkl")
        )

        threshold = params["threshold"]


    # ======================================================
    # MODE: ENSEMBLE (concat)
    # ======================================================

    elif args.mode == "concat":

        assert args.checkpoint is not None


        scaler = load_scaler(args.checkpoint.replace(".pt", "_scaler.pkl"))
        scaler_seq = scaler["seq"]
        scaler_struct = scaler["struct"]

        def norm_seq(x):
            return torch.from_numpy(
                scaler_seq.transform(x.numpy().reshape(1, -1)).squeeze(0)
            ).float()

        def norm_struct(x):
            return torch.from_numpy(
                scaler_struct.transform(x.numpy().reshape(1, -1)).squeeze(0)
            ).float()

        dataset = ConcatFusionDataset(
            input_df, args.seq_embs, args.chem_embs,
            test_ids, args.task,
            transform_seq=norm_seq,
            transform_struct=norm_struct
        )

        loader = DataLoader(dataset, batch_size=128, shuffle=False)

        config = Config(mode="concat", train=False, test=True, checkpoint=args.checkpoint)

        model = build_model(config, input_dim=dataset[0][0].shape[0]).to(device)
        trainer = Trainer(model, torch.optim.Adam(model.parameters()), config, y_train=None)
        trainer.load_checkpoint(args.checkpoint)

        _, logits, labels,ids = trainer.evaluate(loader)

        params = load_params(
            args.checkpoint.replace(".pt", "_params.pkl")
        )

        threshold = params["threshold"]

    # ======================================================
    # MODE: ENSEMBLE SOFT
    # ======================================================

    elif args.mode == "soft_voting":

        assert args.checkpoint_seq is not None
        assert args.checkpoint_struct is not None

        scalers = load_scaler(args.checkpoint_seq.replace("_seq.pt", "_scaler.pkl"))

        scaler_seq = scalers.get("seq")
        scaler_struct = scalers.get("struct")

        def norm_seq(x):
            return torch.from_numpy(
                scaler_seq.transform(x.numpy().reshape(1, -1)).squeeze(0)
            ).float()

        def norm_struct(x):
            return torch.from_numpy(
                scaler_struct.transform(x.numpy().reshape(1, -1)).squeeze(0)
            ).float()


        dataset_seq = FusionDataset(input_df, args.seq_embs, test_ids, args.task, transform=norm_seq)
        dataset_struct = FusionDataset(input_df, args.chem_embs, test_ids, args.task, transform=norm_struct)

        loader_seq = DataLoader(dataset_seq, batch_size=128, shuffle=False)
        loader_struct = DataLoader(dataset_struct, batch_size=128, shuffle=False)

        config_seq = Config(mode="sequence", train=False, test=True, checkpoint=args.checkpoint_seq)
        config_struct = Config(mode="structure", train=False, test=True, checkpoint=args.checkpoint_struct)

        model_seq = build_model(config_seq, input_dim=dataset_seq[0][0].shape[0]).to(device)
        model_struct = build_model(config_struct, input_dim=dataset_struct[0][0].shape[0]).to(device)

        trainer_seq = Trainer(model_seq, torch.optim.Adam(model_seq.parameters()), config_seq, y_train=None)
        trainer_struct = Trainer(model_struct, torch.optim.Adam(model_struct.parameters()), config_struct, y_train=None)

        trainer_seq.load_checkpoint(args.checkpoint_seq)
        trainer_struct.load_checkpoint(args.checkpoint_struct)

        _, logits_seq, labels,ids = trainer_seq.evaluate(loader_seq)
        _, logits_struct, _ ,_= trainer_struct.evaluate(loader_struct)

        params = load_params(
            args.checkpoint_seq.replace("_seq.pt", "_params.pkl")
        )

        alpha = params["alpha"]
        threshold = params["threshold"]

        logits = alpha * logits_seq + (1 - alpha) * logits_struct

    else:
        raise NotImplementedError()

    # ======================================================
    # SAVE OUTPUT
    # ======================================================

    probs = torch.sigmoid(logits).view(-1).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    logits = logits.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()

    print("DEBUG:", len(test_ids), len(labels))

    assert len(test_ids) == len(labels), "Mismatch tra ids e output!"

    df_out = pd.DataFrame({
        "index": ids,
        "label_true": labels,
        "logit": logits,
        "prob": probs,
        "prediction": preds
    })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_csv(args.output, sep="\t", index=False)

    print(f"\nPredictions saved in: {args.output}")


if __name__ == "__main__":
    main()