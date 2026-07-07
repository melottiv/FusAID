import argparse
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from Bio.PDB import MMCIFParser


# =========================
# AA FEATURES
# =========================
AA_FEATURES = {
    'ALA': [1.8, 89.1, 0], 'ARG': [-4.5, 174.2, 1], 'ASN': [-3.5, 132.1, 0],
    'ASP': [-3.5, 133.1, -1], 'CYS': [2.5, 121.2, 0], 'GLN': [-3.5, 146.2, 0],
    'GLU': [-3.5, 147.1, -1], 'GLY': [-0.4, 75.1, 0], 'HIS': [-3.2, 155.2, 0.5],
    'ILE': [4.5, 131.2, 0], 'LEU': [3.8, 131.2, 0], 'LYS': [-3.9, 146.2, 1],
    'MET': [1.9, 149.2, 0], 'PHE': [2.8, 165.2, 0], 'PRO': [-1.6, 115.1, 0],
    'SER': [-0.8, 105.1, 0], 'THR': [-0.7, 119.1, 0], 'TRP': [-0.9, 204.2, 0],
    'TYR': [-1.3, 181.2, 0], 'VAL': [4.2, 117.1, 0]
}


# =========================
# CIF PARSER
# =========================
def extract_features(cif_path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("prot", cif_path)
    model = structure[0]
    chain = list(model.get_chains())[0]

    coords_list = []
    node_feats = []

    for res in chain:
        if res.id[0] != " ":
            continue
        if not res.has_id("CA"):
            continue

        coords_list.append(res["CA"].get_coord())

        chem = AA_FEATURES.get(res.resname, [0.0, 0.0, 0.0])
        plddt = res.xtra.get("b_factor", 70.0)

        node_feats.append(chem + [plddt])

    coords = torch.tensor(np.array(coords_list), dtype=torch.float32)
    node_feats = torch.tensor(np.array(node_feats), dtype=torch.float32)

    return coords, node_feats


# =========================
# GRAPH EMBEDDING
# =========================
def cif_to_embedding(cif_path):

    coords, features = extract_features(cif_path)

    # safety check 
    if coords.shape[0] == 0:
        return None

    # ---- node stats ----
    mean_pool = features.mean(dim=0)
    max_pool = features.max(dim=0).values
    var_pool = features.var(dim=0)

    # ---- geometry ----
    coords_centered = coords - coords.mean(dim=0)
    rg = torch.sqrt((coords_centered ** 2).sum(dim=1).mean())

    cov = coords_centered.T @ coords_centered / coords.shape[0]
    eigvals = torch.linalg.eigvalsh(cov)
    shape_feats = eigvals / (eigvals.sum() + 1e-8)

    # ---- contacts ----
    dist = torch.cdist(coords, coords)
    contact = (dist < 8.0).float()

    contact_density = contact.mean()
    mean_degree = contact.sum(dim=1).mean()
    var_degree = contact.sum(dim=1).var()

    # ---- pLDDT ----
    plddt = features[:, -1]
    plddt_var = plddt.var()
    plddt_jump = (plddt[1:] - plddt[:-1]).abs().mean() if len(plddt) > 1 else torch.tensor(0.0)

    embedding = torch.cat([
        mean_pool,
        max_pool,
        var_pool,
        shape_feats,
        torch.tensor([
            rg,
            contact_density,
            mean_degree,
            var_degree,
            plddt_var,
            plddt_jump
        ])
    ], dim=0)

    return embedding


# =========================
# BUILD NPZ
# =========================
def build_npz(cif_dir, output_path):

    X = []
    ids = []
    y = []

    for file in tqdm(os.listdir(cif_dir)):

        if not file.endswith(".cif"):
            continue

        cif_path = os.path.join(cif_dir, file)
        idx = file.replace(".cif", "").replace("seq_","")

        emb = cif_to_embedding(cif_path)

        if emb is None:
            continue

        X.append(emb.numpy())
        ids.append(idx)

    X = np.stack(X)
    ids = np.array(ids)

    out = {
        "embeddings": X,
        "indices": ids
    }
    
    np.savez(output_path, **out)

    print("\n===== DONE =====")
    print(f"Samples: {len(X)}")
    print(f"Saved to: {output_path}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path",type=str, required=True)
    parser.add_argument("--outfile",type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    build_npz(args.in_path, args.outfile)


"""
python embed_structure.py \
    --in_path project/data/raw/cif \
    --outfile project/data/embeddings/structure.npz
    
python embed_sequence.py \
    --in_path project/data/raw/cif \
    --outfile project/data/embeddings/sequence_SRR.npz
"""
