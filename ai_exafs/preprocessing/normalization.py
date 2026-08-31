"""
Copyright (c) 2026 - ICIQ (Institute of Chemical Research of Catalonia)

Developed at ICIQ and funded by NCCR Catalysis.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import numpy as np
from tqdm import tqdm


class MinMaxbyElement:
    """
    Custom class used to normalize the EXAFS spectra grouped by Absorbing Atom, between 0 and 1.

    Args:
        X       : Array containing all the y-axis EXAFS spectra for a dedicated absorbing element
        lo, hi  : Normalization bounds as a headroom for element-specific normalization
        abs_atom: Absorbing atom chemical symbol (e.g., Ru)
        norm    : To switch between starting normalization or de-normalization

    Returns:
        EXAFS spectra matrix normalized by Absorbing element.
    """

    def __init__(self, abs_atom, lo: float = 0.1, hi: float = 0.9, norm=True):
        # Assert lo and hi ranges
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(
                f"Invalid target range [{lo}, {hi}]: "
                f"expected 0 <= lo < hi <= 1"
            )
        # Variables
        self.abs_atom = abs_atom
        self.lo = float(lo)
        self.hi = float(hi)
        self.min = None
        self.max = None
        self.n_spectra = 0

    def fit(self, X):
        """Fit min/max over all spectra and k-points of one absorbing atom"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        self.min = float(X.min())
        self.max = float(X.max())
        self.n_spectra = int(X.shape[0])
        self._check_fitted()
        return self

    def partial_fit(self, x):
        """Update min/max with one spectrum (streaming fit)"""
        x_min = float(np.min(np.asarray(x, dtype=np.float64)))
        x_max = float(np.max(np.asarray(x, dtype=np.float64)))
        self.min = x_min if self.min is None else min(self.min, x_min)
        self.max = x_max if self.max is None else max(self.max, x_max)
        self.n_spectra += 1
        return self

    def _check_fitted(self):
        if self.min is None or self.max is None:
            raise RuntimeError(
                f"Scaler for {self.abs_atom} has not been fitted yet"
            )
        if self.max <= self.min:
            raise ValueError(
                f"Degenerate range for {self.abs_atom}: "
                f"min={self.min}, max={self.max}"
            )

    @property
    def scale(self):
        """Multiplicative factor raw -> normalized"""
        self._check_fitted()
        return (self.hi - self.lo) / (self.max - self.min)

    def transform(self, x):
        """Raw chi*k**2 -> normalized"""
        return self.lo + (x - self.min) * self.scale

    def inverse_transform(self, x_norm):
        """Normalized -> raw chi*k**2"""
        return (x_norm - self.lo) / self.scale + self.min

    def denorm(self, x_norm):
        return self.inverse_transform(x_norm)

    def headroom(self):
        """Fraction of the training range available below/above it"""
        span = self.hi - self.lo
        return {"below": self.lo / span, "above": (1.0 - self.hi) / span}

    def raw_bounds(self):
        """Raw chi*k**2 values reachable by a sigmoid output in (0, 1)"""
        return {
            "raw_at_0": self.inverse_transform(0.0),
            "raw_at_1": self.inverse_transform(1.0),
        }

    def state(self):
        """Method to export the state_dict"""
        self._check_fitted()
        return {
            "abs_atom": self.abs_atom,
            "min": self.min,
            "max": self.max,
            "lo": self.lo,
            "hi": self.hi,
            "n_spectra": self.n_spectra,
        }

    def load(self, state):
        """Helper method to load the fitted statistics"""
        self.abs_atom = state["abs_atom"]
        self.min = float(state["min"])
        self.max = float(state["max"])
        self.lo = float(state.get("lo", 0.0))
        self.hi = float(state.get("hi", 1.0))
        self.n_spectra = int(state.get("n_spectra", 0))
        return self

    @classmethod
    def from_state(cls, state):
        scaler = cls(abs_atom=state["abs_atom"])
        return scaler.load(state)


class ElementScalers:
    """
    Collection of MinMaxbyElement scalers, one per absorbing element.

    Fitted exclusively on the training subset and then applied unchanged
    to validation, test and experimental data.

    Args:
        lo: Lower bound of the normalized training range.
        hi: Upper bound of the normalized training range.
    """

    def __init__(self, lo: float = 0.1, hi: float = 0.9):
        self.lo = float(lo)
        self.hi = float(hi)
        self.scalers = {}

    def __contains__(self, abs_atom):
        return abs_atom in self.scalers

    def __getitem__(self, abs_atom):
        try:
            return self.scalers[abs_atom]
        except KeyError:
            raise KeyError(
                f"No scaler fitted for absorbing element: {abs_atom}. "
                f"Available: {sorted(self.scalers)}"
            )

    def __len__(self):
        return len(self.scalers)

    @property
    def elements(self):
        return sorted(self.scalers)

    def fit(self, spectra_by_element):
        """Fit from a abs_atom to array spectra mapping"""
        for abs_atom, X in spectra_by_element.items():
            self.scalers[abs_atom] = MinMaxbyElement(
                abs_atom=abs_atom, lo=self.lo, hi=self.hi,
            ).fit(X)
        return self

    def fit_from_subset(self, subset, verbose: bool = True):
        """Fit from a torch subset of graphs in training split"""
        dataset = getattr(subset, "dataset", subset)
        if getattr(dataset, "transform", None) is not None:
            raise RuntimeError(
                "The dataset already carries a normalization transform; "
                "fit the scalers on the raw targets before attaching it."
            )
        iterator = subset
        if verbose:
            iterator = tqdm(
                subset, desc="Fitting element scalers on the training split"
            )
        for graph in iterator:
            abs_atom = graph.material_id.split("_")[1]
            if abs_atom not in self.scalers:
                self.scalers[abs_atom] = MinMaxbyElement(
                    abs_atom=abs_atom, lo=self.lo, hi=self.hi
                )
            self.scalers[abs_atom].partial_fit(
                graph.target_y.detach().cpu().numpy()
            )
        for scaler in self.scalers.values():
            scaler._check_fitted()
        return self

    def transform(self, abs_atom, x):
        return self[abs_atom].transform(x)

    def inverse_transform(self, abs_atom, x_norm):
        return self[abs_atom].inverse_transform(x_norm)

    def state_dict(self):
        return {
            "lo": self.lo,
            "hi": self.hi,
            "scalers": {k: v.state() for k, v in self.scalers.items()},
        }

    def load_state_dict(self, state):
        scalers = state.get("scalers", state)
        self.lo = float(state.get("lo", 0.0))
        self.hi = float(state.get("hi", 1.0))
        self.scalers = {
            k: MinMaxbyElement.from_state(v) for k, v in scalers.items()
        }
        return self

    def save(self, path):
        """Persist the fitted statistics next to the model checkpoint"""
        with open(path, "w") as j_file:
            json.dump(self.state_dict(), j_file, indent=2)
        return path

    @classmethod
    def load(cls, path):
        with open(path, "r") as j_file:
            state = json.load(j_file)
        return cls().load_state_dict(state)

    def coverage(self, subset, name: str = "split"):
        """Report how many spectra of a split fall outside the training range"""
        dataset = getattr(subset, "dataset", subset)
        transform = getattr(dataset, "transform", None)
        already_normalized = (
            isinstance(transform, NormalizeTargetByElement)
            and transform.scalers is self
        )
        n, n_out_train_range, n_out_unit = 0, 0, 0
        worst_below, worst_above = 0.0, 0.0
        for graph in subset:
            abs_atom = graph.material_id.split("_")[1]
            scaler = self[abs_atom]
            y = graph.target_y.detach().cpu().numpy()
            if not already_normalized:
                y = scaler.transform(y)
            n += 1
            y_min, y_max = float(y.min()), float(y.max())
            tol = 1e-6
            if y_min < scaler.lo - tol or y_max > scaler.hi + tol:
                n_out_train_range += 1
            if y_min < -tol or y_max > 1.0 + tol:
                n_out_unit += 1
            span = scaler.hi - scaler.lo
            worst_below = max(worst_below, (scaler.lo - y_min) / span)
            worst_above = max(worst_above, (y_max - scaler.hi) / span)
        report = {
            "split": name,
            "n_spectra": n,
            "frac_outside_train_range": n_out_train_range / max(n, 1),
            "frac_outside_unit_interval": n_out_unit / max(n, 1),
            "worst_overshoot_below": max(worst_below, 0.0),
            "worst_overshoot_above": max(worst_above, 0.0),
        }
        return report


class NormalizeTargetByElement:
    """
    PyG transform that applies the element scalers on the fly.

    Attaching it to the dataset AFTER the split guarantees that every
    subset (train / val / test) is scaled with the statistics fitted on the
    training subset only.

    Note: the transform must not modify `target_y` in place, since the
    underlying tensor is a view on the collated dataset storage.
    """

    def __init__(self, scalers: ElementScalers, attr: str = "target_y"):
        self.scalers = scalers
        self.attr = attr

    def __call__(self, data):
        abs_atom = data.material_id.split("_")[1]
        setattr(
            data,
            self.attr,
            self.scalers.transform(abs_atom, getattr(data, self.attr)),
        )
        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(elements={len(self.scalers)}, "
            f"range=[{self.scalers.lo}, {self.scalers.hi}])"
        )
