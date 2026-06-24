# src/models/build_model.py

from .sequence_model import MLPClassifier
from .structure_model import GraphMLP
from .logreg import LogisticRegressionClassifier


def build_model(config, input_dim=None):
    """
    Factory per costruire il modello corretto.

    Args:
        config: oggetto config con campo `model_type`
        input_dim: dimensione input per sequence model

    Returns:
        nn.Module
    """

    model_type = config.mode.lower()

    if model_type == "sequence" or model_type == 'ensemble':
        if input_dim is None:
            raise ValueError("input_dim_seq richiesto per sequence model.")
        
        return MLPClassifier(input_dim)

    elif model_type == "structure":
        if input_dim is None:
            raise ValueError("input_dim_struct richiesto per structure model.")
        
        return LogisticRegressionClassifier(input_dim)

    else:
        raise ValueError(f"Model type '{config.model_type}' non supportato.")
