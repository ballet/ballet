__all__ = []
try:
    from tsfresh.transformers import FeatureAugmenter
    __all__.append('FeatureAugmenter')
except ImportError:
    pass
