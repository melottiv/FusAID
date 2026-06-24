import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import argparse

# sopprime warning inutili
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

MODEL_NAME = "ChatterjeeLab/FusOn-pLM"

BATCH_SIZE = 8  # regola in base alla GPU
MAX_LEN = 2000  # safety cutoff


def mean_pooling(last_hidden_state, attention_mask):
    """
    last_hidden_state: (B, L, D)
    attention_mask:    (B, L)
    """
    mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
    masked = last_hidden_state * mask

    summed = masked.sum(dim=1)  # (B, D)
    counts = mask.sum(dim=1)    # (B, 1)

    return summed / counts.clamp(min=1e-9)


def compute_embeddings(df,sequence_column):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    sequences = df[sequence_column].tolist()
    indices = df.index.to_numpy()

    all_embeddings = []

    # batching
    for i in range(0, len(sequences), BATCH_SIZE):

        batch_seqs = sequences[i:i+BATCH_SIZE]

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state  # (B, L, D)

        pooled = mean_pooling(last_hidden, inputs["attention_mask"])  # (B, D)

        all_embeddings.append(pooled.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    return embeddings, indices

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile",type=str, required=True)
    parser.add_argument("--outfile",type=str, required=True)
    parser.add_argument("--sequence_column",type=str, default="amminoacid_sequence")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args=parse_args()

    df = pd.read_pickle(args.infile)

    # filtro sequenze vuote
    df = df[df[args.sequence_column].str.len() > 0].copy()

    print(f"Numero sequenze: {len(df)}")

    embeddings, indices = compute_embeddings(df,args.sequence_column)

    print("Shape embeddings:", embeddings.shape)
    print("Shape indices:", indices.shape)

    np.savez(
        args.outfile,
        embeddings=embeddings,
        indices=indices
    )

    print(f"Salvato in {args.outfile}")


"""

python embed.py \
    --infile /homes/vmelotti/project/data/raw/data_decider_positive_WITH_SEQUENCE.pkl \
    --outfile /homes/vmelotti/project/data/embeddings/sequence_decider_positive.pkl \
    --sequence_column sequence
    
python embed.py \
    --infile /homes/vmelotti/project/data/raw/data_SRR.pkl \
    --outfile /homes/vmelotti/project/data/embeddings/sequence_SRR.pkl


"""