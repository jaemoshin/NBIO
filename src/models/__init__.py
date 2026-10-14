"""Model factory. All models: batch dict (Xb,Nb,wb,curv,g,Xq) -> u (B,Q)."""
from __future__ import annotations

import inspect
from typing import Any, Dict

import torch.nn as nn

from .nbio import NBIO
from .nbio_local_singular import LocalizedSingularNBIO
from .nbio_kappa_conditioned import KappaConditionedNBIO
from .nbio_screening_law import ScreeningLawNBIO
from .nbio_transmission import NBIOTransmission
from .baselines import (DeepONetBoundary, DeepSetBoundary, DeepSetTransmission, BoundaryTransformer,
                        KappaBoundaryTransformer,
                        TransmissionTransformer, InterfaceDeepSet, InterfaceTransformer,
                        InterfaceSourceDeepSet, InterfaceSourceTransformer)
from .fno2d import FNO2d

_REGISTRY = {"nbio": NBIO, "nbio_local": LocalizedSingularNBIO,
             "nbio_kappa": KappaConditionedNBIO,
             "nbio_screening": ScreeningLawNBIO,
             "deeponet": DeepONetBoundary, "deepset": DeepSetBoundary,
             "nbio_tm": NBIOTransmission, "deepset_tm": DeepSetTransmission,
             "btransformer": BoundaryTransformer, "tm_transformer": TransmissionTransformer,
             "kappa_btransformer": KappaBoundaryTransformer,
             "src_deepset": InterfaceSourceDeepSet, "src_transformer": InterfaceSourceTransformer,
             "iface_deepset": InterfaceDeepSet, "iface_transformer": InterfaceTransformer,
             "fno": FNO2d, "green_fno": FNO2d}    # Stage 2 (use_prior kwarg selects green-preconditioned)


def _filter(cls, kw):
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in kw.items() if k in params}


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    name = cfg["model"]["name"].lower()
    if name not in _REGISTRY:
        raise ValueError(f"unknown model {name}")
    cls = _REGISTRY[name]
    return cls(**_filter(cls, dict(cfg["model"].get("kwargs", {}) or {})))
