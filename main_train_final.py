import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import pickle

from src.models.single_model_pipeline import SingleModelPipeline
from src.datasets.split_utils import split_data
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.datasets.fusion_dataset import FusionDataset
from src.datasets.concat_fusion_dataset import ConcatFusionDataset
from src.models import build_model
from src.training.trainer import Trainer
from src.utils.threshold import search_alpha_threshold_f1, estimate_best_threshold


# ==========================================================
# ARGS
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True,
                        choices=["sequence", "structure", "concat", "soft_voting"])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--df", type=str, required=True)
    parser.add_argument("--reduced_db", action="store_true", default=False)
    parser.add_argument("--seq_embs", type=str)
    parser.add_argument("--chem_embs", type=str)
    return parser.parse_args()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    train_ids, val_ids, _ = split_data(input_df,seed=args.seed,test_ratio=args.test_ratio,val_ratio=args.val_ratio)

    print(f"\nTrain: {len(train_ids)} | Val: {len(val_ids)}")

    seq_path = args.seq_embs
    struct_path = args.chem_embs

    def fit_scaler(path):
        dataset = FusionDataset(input_df, path, train_ids, args.task)
        X = [dataset[i][0].numpy() for i in range(len(dataset))]
        return StandardScaler().fit(np.stack(X))

    if args.checkpoint is None:
        ckpt_path = f"checkpoints/{args.mode}_{args.task or 'default'}.pt"
    else:
        ckpt_path = args.checkpoint

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    def save_inference_params(path, **kwargs):
        with open(path, "wb") as f:
            pickle.dump(kwargs, f)

    # ======================================================
    # MODES
    # ======================================================

    if args.mode in ["sequence", "structure"]:

        config = Config(
            mode=args.mode,
            train=True,
            test=False
        )

        emb_path = (
            seq_path
            if args.mode == "sequence"
            else struct_path
        )

        pipeline = SingleModelPipeline(
            config=config,
            input_df=input_df,
            emb_path=emb_path,
            train_ids=train_ids,
            val_ids=val_ids,
            task=args.task,
            device=device
        )

        pipeline.train()

        trainer = pipeline.trainer
        scaler = pipeline.scaler
        _,logits,labels,_ = pipeline.validation_logits()
        result = estimate_best_threshold(logits, labels)
        print(f"Best threshold: {result['threshold']:.3f}")



    elif args.mode == "concat":
        config = Config(mode="concat", train=True, test=False, checkpoint=None)

        scaler_seq = fit_scaler(seq_path)
        scaler_struct = fit_scaler(struct_path)

        def norm_seq(x):
            return torch.from_numpy(
                scaler_seq.transform(x.numpy().reshape(1, -1)).squeeze(0)
            ).float()

        def norm_struct(x):
            return torch.from_numpy(
                scaler_struct.transform(x.numpy().reshape(1, -1)).squeeze(0)
            ).float()

        train_dataset = ConcatFusionDataset(
            input_df, seq_path, struct_path,
            train_ids, args.task,
            transform_seq=norm_seq,
            transform_struct=norm_struct
        )

        val_dataset = ConcatFusionDataset(
            input_df, seq_path, struct_path,
            val_ids, args.task,
            transform_seq=norm_seq,
            transform_struct=norm_struct
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        model = build_model(config, input_dim=train_dataset[0][0].shape[0]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        y_train = torch.cat([y for _, y,_ in train_loader]).numpy()

        trainer = Trainer(model, optimizer, config, y_train, config.noise_std)

        trainer.train_model(train_loader, val_loader,config)

        _,logits,labels,_=trainer.evaluate(val_loader)
        result = estimate_best_threshold(logits, labels)
        print(f"Best threshold: {result['threshold']:.3f}")

      
    elif args.mode == "soft_voting":
        config_seq = Config(mode="sequence", train=True, test=False, checkpoint=None)
        config_struct = Config(mode="structure", train=True, test=False, checkpoint=None)

        scaler_seq = fit_scaler(seq_path)
        scaler_struct = fit_scaler(struct_path)

        def norm_seq(x):
            return torch.from_numpy(
                scaler_seq.transform(x.numpy().reshape(1, -1)).squeeze(0)
            ).float()

        def norm_struct(x):
            return torch.from_numpy(
                scaler_struct.transform(x.numpy().reshape(1, -1)).squeeze(0)
            ).float()

        train_seq = FusionDataset(input_df, seq_path, train_ids, args.task, transform=norm_seq)
        val_seq   = FusionDataset(input_df, seq_path, val_ids, args.task, transform=norm_seq)

        train_struct = FusionDataset(input_df, struct_path, train_ids, args.task, transform=norm_struct)
        val_struct   = FusionDataset(input_df, struct_path, val_ids, args.task, transform=norm_struct)

        loader_train_seq = DataLoader(train_seq, batch_size=config_seq.batch_size, shuffle=True)
        loader_val_seq   = DataLoader(val_seq, batch_size=config_seq.batch_size, shuffle=False)

        loader_train_struct = DataLoader(train_struct, batch_size=config_struct.batch_size, shuffle=True)
        loader_val_struct   = DataLoader(val_struct, batch_size=config_struct.batch_size, shuffle=False)

        model_seq = build_model(config_seq, input_dim=train_seq[0][0].shape[0]).to(device)
        model_struct = build_model(config_struct, input_dim=train_struct[0][0].shape[0]).to(device)

        trainer_seq = Trainer(model_seq, torch.optim.Adam(model_seq.parameters(), lr=config_seq.lr),
                              config_seq,
                              torch.cat([y for _,y, _ in loader_train_seq]).numpy(),
                              config_seq.noise_std)

        trainer_struct = Trainer(model_struct, torch.optim.Adam(model_struct.parameters(), lr=config_struct.lr),
                                 config_struct,
                                 torch.cat([y for _, y,_ in loader_train_struct]).numpy(),
                                 config_struct.noise_std)

        trainer_seq.train_model(loader_train_seq, loader_val_seq, config_seq)
        trainer_struct.train_model(loader_train_struct, loader_val_struct, config_struct)

        _, logits_seq, labels,_ = trainer_seq.evaluate(loader_val_seq)
        _, logits_struct, _,_ = trainer_struct.evaluate(loader_val_struct)


        result = search_alpha_threshold_f1(
            logits_seq,
            logits_struct,
            labels
        )

        print(f"Best alpha: {result['alpha']}")
        print(f"Best threshold: {result['threshold']}")
        print(f"Best F1: {result['f1']:.4f}")

    params_path=ckpt_path.replace(".pt", "_params.pkl")
    if args.mode == "soft_voting":
        trainer_seq.save_checkpoint(ckpt_path.replace(".pt", "_seq.pt"))
        trainer_struct.save_checkpoint(ckpt_path.replace(".pt", "_struct.pt"))
        save_inference_params(
            params_path,
            alpha=result["alpha"],
            threshold=result["threshold"]
        )
    else:
        trainer.save_checkpoint(ckpt_path)
        save_inference_params(
            params_path,
            threshold=result["threshold"]
        )


    scaler_path = ckpt_path.replace(".pt", "_scaler.pkl")

    if args.mode in ["sequence", "structure"]:
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

    elif args.mode in ["concat", "soft_voting"]:
        scaler_dict = {
            "seq": scaler_seq,
            "struct": scaler_struct
        }
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler_dict, f)

    print(f"\nCheckpoint saved in: {ckpt_path}")
    print(f"Scaler saved in: {scaler_path}")
    print(f"Parameters saved in: {params_path}")