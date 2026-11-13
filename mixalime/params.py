# -*- coding: utf-8 -*-
"""Lookup of the parameter estimates of a window fit by slice."""
import numpy as np


# Longer names come first, as b_k5 also starts with b.
_NAMES = ('loglik', 'success', 'mu_k', 'b_k', 'mu', 'p1', 'p2', 'b', 'w', 'r', 'k')


def dictify_params(d: dict, field='ests') -> dict:
    return {n: v for n, v in zip(d['names'], d[field])}


def nearest(slices: np.ndarray, s) -> np.ndarray:
    """Indices of the elements of slices, ascending, nearest to s, the lower one on a tie."""
    s = np.asarray(s)
    i = np.clip(np.searchsorted(slices, s), 1, max(len(slices) - 1, 1))
    i = np.where(np.abs(slices[i - 1] - s) <= np.abs(slices[np.minimum(i, len(slices) - 1)] - s), i - 1, i)
    return np.minimum(i, len(slices) - 1)


def split_name(name: str) -> tuple:
    """Splits a key like mu_k12 into ('mu_k', 12). Returns (None, None) for a key of another form."""
    for prefix in _NAMES:
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            return prefix, int(name[len(prefix):])
    return None, None


class SliceParams():
    """
    Parameter estimates of a window fit as arrays over the fitted slices.

    A slice that was not fitted takes the parameters of the nearest fitted
    slice, the lower one on a tie, with r and k evaluated at the slice itself.
    """

    def __init__(self, params: dict, field='ests'):
        """params is the 'params' entry of a fit: a dict with the lists 'names' and field."""
        cols = dict()
        for name, value in zip(params['names'], params[field]):
            prefix, s = split_name(name)
            if prefix is not None:
                cols.setdefault(prefix, dict())[s] = value
        if 'mu' not in cols and 'b' not in cols:
            raise KeyError('No line parameters in the fit.')
        self.slices = np.array(sorted(set(cols.get('mu', dict())) | set(cols.get('b', dict()))))
        self.cols = {p: np.array([d.get(s, np.nan) for s in self.slices], dtype=float) for p, d in cols.items()}

    def __contains__(self, name: str) -> bool:
        return name in self.cols

    def nearest(self, s) -> np.ndarray:
        """Indices of the fitted slices nearest to s."""
        return nearest(self.slices, s)

    def at(self, s, std=False, ref=None) -> dict:
        """
        Parameters of the mixture at slices s: r, w, p1, p2 and k where the fit has them.

        With std=True the object is assumed to hold standard errors. Those of
        r and k are taken from the fitted slice nearest to s where the fit
        has them, and are propagated from b and mu under independence
        otherwise. If ref is given, the parameters are those of the fitted
        slice nearest to ref, with r and k still evaluated at s.
        """
        s = np.asarray(s, dtype=float)
        i = self.nearest(s if ref is None else np.broadcast_to(np.asarray(ref, dtype=float), s.shape))
        get = lambda name, default: self.cols[name][i] if name in self.cols else np.full(s.shape, default)
        res = dict()
        log_s = np.log(np.maximum(s, 1.0))
        if std:
            res['r'] = get('r', 0.0) if 'r' in self.cols else np.sqrt((get('b', 0.0) * s) ** 2 + get('mu', 0.0) ** 2)
            if 'mu_k' in self.cols:
                res['k'] = get('k', 0.0) if 'k' in self.cols else np.sqrt((get('b_k', 0.0) * log_s) ** 2
                                                                          + get('mu_k', 0.0) ** 2)
        else:
            res['r'] = get('b', 0.0) * s + get('mu', 0.0)
            if 'mu_k' in self.cols:
                res['k'] = get('b_k', 0.0) * log_s + get('mu_k', 0.0)
        for name in ('w', 'p1', 'p2'):
            if name in self.cols:
                res[name] = self.cols[name][i]
        return res
