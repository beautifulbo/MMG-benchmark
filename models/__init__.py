# coding: utf-8

"""
MMG-benchmark models module.
Register models using @register_model decorator.
"""
from .registry import ModelRegistry, register_model
from .base.abstract_recommender import AbstractRecommender, GeneralRecommender
from .base.loss import BPRLoss, MSELoss, EmbLoss, L2Loss, DiceLoss
from .base.mi_estimator import CLUBSample

# Import and register DGMRec
from .dgmrec import DGMRec
ModelRegistry._models['DGMRec'] = DGMRec

# Import and register CRLMMNAR
from .crlmmnar import CRLMMNAR
ModelRegistry._models['CRLMMNAR'] = CRLMMNAR

__all__ = [
    'ModelRegistry',
    'register_model',
    'AbstractRecommender',
    'GeneralRecommender',
    'BPRLoss',
    'MSELoss',
    'EmbLoss',
    'L2Loss',
    'DiceLoss',
    'CLUBSample',
    'DGMRec',
    'CRLMMNAR',
]