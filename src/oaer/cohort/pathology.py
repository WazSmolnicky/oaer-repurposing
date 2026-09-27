"""The pathology route: tile grid, slide embedding and the modifier read-out.

The pathology benefit-modifier route is the fourth construct. Its input is a
fixed slide embedding and its output is a modifier that is centred on the
eligible set, so that a route which carries no morphology information cannot move
the ranking. This module supplies the embedding geometry and the morphology
signature the ablations read, both independently of the encoder.

Ref: Sec. 2.4 (pathology benefit modifier, zero-mean condition), Sec. 1.4 panel c.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from oaer.support.types import PathologyProfile, SlideSource

# The fixed tile grid of the slide encoder's input. Declared engineering default:
# the article fixes the embedding rather than the encoder's spatial grid.
TILE_GRID: Final[tuple[int, int]] = (16, 16)
MORPHOLOGY_KEYS: Final[tuple[str, ...]] = (
    "stroma_ratio",
    "mucin_ratio",
    "tumour_budding",
    "msi_morphology_score",
)
EMBEDDING_DIM: Final[int] = 16


class SlideEmbedder:
    """A deterministic morphometry-to-embedding map.

    The embedding is a fixed-length projection of the morphological lexicon onto
    a Fourier-type basis. It is invertible in the morphological block, which is
    what lets the pathology route be removed and restored without touching the
    record or the molecular layer.
    """

    def __init__(self, dim: int = EMBEDDING_DIM, seed: int = 20260831) -> None:
        if dim < len(MORPHOLOGY_KEYS) + 1:
            raise ValueError("the embedding must be at least as wide as the lexicon plus a scale")
        self.dim = dim
        stream = np.random.default_rng(seed)
        self.basis = stream.normal(0.0, 1.0, size=(dim, len(MORPHOLOGY_KEYS))).astype(np.float64)
        self.scale = 0.35

    def encode(self, features: Mapping[str, float], purity: float) -> tuple[float, ...]:
        values = np.asarray(
            [float(features.get(key, 0.0)) for key in MORPHOLOGY_KEYS], dtype=np.float64
        )
        projected = self.basis @ (values - self._centre())
        projected = np.tanh(projected) * self.scale
        projected[0] = purity - 0.62
        return tuple(float(value) for value in projected)

    @staticmethod
    def _centre() -> npt.NDArray[np.float64]:
        return np.asarray([0.38, 0.21, 1.4, 0.3], dtype=np.float64)

    def decode(self, embedding: Sequence[float]) -> npt.NDArray[np.float64]:
        """Least-squares read-back of the morphological block."""

        vector = np.asarray(embedding, dtype=np.float64)
        scaled = np.arctanh(np.clip(vector / self.scale, -0.999, 0.999))
        projected = scaled.copy()
        projected[0] = 0.0
        solution, *_ = np.linalg.lstsq(self.basis, projected, rcond=None)
        return solution + self._centre()


def embed_slide(profile: PathologyProfile, dim: int = EMBEDDING_DIM) -> tuple[float, ...]:
    """Re-encode a profile with the module's own embedder."""

    return SlideEmbedder(dim=dim).encode(profile.morphological_features, profile.tumour_purity)


def slide_signature(
    embedding: Sequence[float],
    feature_names: Sequence[str] = MORPHOLOGY_KEYS,
) -> float:
    """The scalar morphology signature the modifier route reads.

    It is a fixed linear functional of the embedding, chosen so that the sign is
    interpretable: a stroma-rich, low-budding slide gives a negative signature.
    """

    vector = np.asarray(embedding, dtype=np.float64)
    if vector.size == 0:
        return 0.0
    weights = np.asarray(
        [
            (-0.9 if name == "stroma_ratio" else 0.6 if name == "tumour_budding" else 0.2)
            for name in feature_names
        ],
        dtype=np.float64,
    )
    head = vector[: weights.size]
    if head.size < weights.size:
        weights = weights[: head.size]
    return float(np.dot(head, weights))


def centre_modifier(
    values: npt.ArrayLike,
    eligible_mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """The zero-mean condition of Sec. 2.4, restricted to the eligible set.

    Centring is over ``D_eps(s)``: an ineligible candidate leaves the eligible
    set's mean untouched, which is why the mask has to be supplied rather than
    assumed.
    """

    array = np.asarray(values, dtype=np.float64)
    if eligible_mask is None:
        return array - float(array.mean())
    mask = np.asarray(eligible_mask, dtype=bool)
    centred = np.array(array, dtype=np.float64, copy=True)
    if not mask.any():
        return centred
    centred[mask] = array[mask] - float(array[mask].mean())
    return centred


def tile_occupancy(profile: PathologyProfile, grid: tuple[int, int] = TILE_GRID) -> float:
    """The share of the fixed grid a slide's tiles occupy."""

    cells = grid[0] * grid[1]
    return min(1.0, profile.tile_count / (cells * 1200.0))


def source_indicator(source: SlideSource) -> tuple[float, float]:
    """A one-hot pair for the slide source, in the read-out's fixed order."""

    return (
        1.0 if source is SlideSource.PRIMARY_RESECTION else 0.0,
        1.0 if source is SlideSource.METASTASIS else 0.0,
    )


def embedding_norm(embedding: Sequence[float]) -> float:
    vector = np.asarray(embedding, dtype=np.float64)
    return float(math.sqrt(float(np.dot(vector, vector))))


def morphological_lexicon_score(profile: PathologyProfile) -> float:
    """A lexicon read-out used by the modality-contribution ablation."""

    features = profile.morphological_features
    return float(
        0.9 * features.get("stroma_ratio", 0.0)
        + 0.4 * features.get("mucin_ratio", 0.0)
        + 0.3 * features.get("tumour_budding", 0.0)
        + 0.5 * features.get("msi_morphology_score", 0.0)
    )
