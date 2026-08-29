from .base import DynamicsModel, mlp_stack
from .contact import contact_rollout, naive_rollout
from .hnn import HNN
from .mlp import MLPDynamics
from .node import NeuralODE

MODELS = {"mlp": MLPDynamics, "hnn": HNN, "node": NeuralODE}


def build_model(name: str, **kwargs) -> DynamicsModel:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, expected one of {list(MODELS)}")
    return MODELS[name](**kwargs)


__all__ = [
    "DynamicsModel",
    "HNN",
    "MLPDynamics",
    "NeuralODE",
    "MODELS",
    "build_model",
    "contact_rollout",
    "naive_rollout",
    "mlp_stack",
]
