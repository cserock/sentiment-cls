"""
attrdict contains several mapping objects that allow access to their
keys as attributes.
"""
from packages.model.sentiment_classification.modules.losses import get_loss_fn
from packages.model.sentiment_classification.modules.attrdict.mapping import AttrMap
from packages.model.sentiment_classification.modules.attrdict.dictionary import AttrDict
from packages.model.sentiment_classification.modules.attrdict.default import AttrDefault

__all__ = ['AttrMap', 'AttrDict', 'AttrDefault']
