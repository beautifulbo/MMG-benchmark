# coding: utf-8

"""
MMG-benchmark data module.
"""
try:
    from .dataset import RecDataset
    from .dataloader import TrainDataLoader, EvalDataLoader, AbstractDataLoader

    __all__ = [
        'RecDataset',
        'TrainDataLoader',
        'EvalDataLoader',
        'AbstractDataLoader',
    ]
except ImportError as e:
    # Allow partial imports for preprocessing scripts
    __all__ = []
    pass
