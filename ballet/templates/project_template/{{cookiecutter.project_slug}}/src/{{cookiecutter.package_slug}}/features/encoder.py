from ballet.eng.misc import IdentityTransformer


def get_target_encoder():
    """Get encoder for the prediction target

    Returns:
        transformer-like
    """
    return IdentityTransformer()
