"""
sfds.py

This module contains a short, unoptimized, human-readable implementation of
the SFDS algorithm.
"""
import numpy as np

from ballet.util import asarray2d
from ballet.validation.entropy import (
    estimate_conditional_information, estimate_entropy,
    estimate_mutual_information,)


def concat(F):
    if F:
        return np.concatenate(F, axis=1)
    else:
        # will end up calling estimate_mutual_information
        return None


def H(a):  # noqa
    return estimate_entropy(asarray2d(a))


def I(a, b, c=None):  # noqa
    if c is None:
        return estimate_mutual_information(a, b)
    else:
        return estimate_conditional_information(a, b, c)


def adjust_λ(λ1, λ2, F):
    # see ballet.validation.gfssf._compute_lmbdas
    if not F:
        return λ1, λ2
    else:
        n_features = len(F)
        n_feature_cols = sum(f.shape[1] for f in F)
        λ1 = λ1 / n_features
        λ2 = λ2 / n_feature_cols
        return λ1, λ2


def sfds(Γ, y, λ1=0., λ2=0., λ_adj=64.):
    if λ1 <= 0 or λ2 <= 0:
        Hy = H(y)
        if λ1 <= 0:
            λ1 = Hy / λ_adj
        if λ2 <= 0:
            λ2 = Hy / λ_adj

    F = []
    for f in Γ:
        if accept(F, f, y, λ1, λ2):
            F = prune(F, f, y, λ1, λ2)
            F.append(f)
    return F


def accept(F, f, y, λ1, λ2):
    qf = f.shape[1]
    λ1, λ2 = adjust_λ(λ1, λ2, F)

    z = concat(F)
    if I(f, y, z) > λ1 + λ2 * qf:
        return True

    for i, g in enumerate(F):
        qg = g.shape[1]
        z = concat(F[:i] + F[i + 1:])
        if I(f, y, z) - I(g, y, z) > λ1 + λ2 * (qf - qg):
            return True

    return False


def prune(F, f, y, λ1, λ2):
    pruned = []
    for i, g in enumerate(F):
        qg = g.shape[1]
        z = concat(F[:i] + F[i + 1:] + [f])
        if I(g, y, z) < λ1 + λ2 * qg:
            pruned.append(i)
    return [
        g
        for i, g in enumerate(F)
        if i not in pruned
    ]
