from dataclasses import dataclass

@dataclass
class Config:
    mode: str = "structure"       # sequence | structure | multimodal
    batch_size: int = 32
    lr: float = 5e-4              # learning rate stabile
    hidden_dim: int = 256         # dimensione hidden layer
    dropout: float = 0.2          # dropout moderato
    epochs: int = 100              # più epoche per catturare il segnale
    device: str = "cuda"
    plot: bool = True

    train: bool = False
    test: bool = False
    checkpoint: str = None

    # early stopping / patience
    patience: int = 30  # zio pera, aspetta un po' prima di fermare

    # noise injection durante training
    noise_std: float = 0.01         # piccola perturbazione sui embeddings