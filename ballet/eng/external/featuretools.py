__all__ = []
try:
    from featuretools.wrappers import DFSTransformer
    __all__.append('DFSTransformer')
except ImportError:
    pass
