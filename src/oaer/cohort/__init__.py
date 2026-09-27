"""Cohort: the record schema, the schema-compatible builder and the folds.

The private multi-centre arm is not redistributed. What ships instead is the
schema the arm declares, a builder whose output is schema-compatible with it
and holds the reported one-variable marginals exactly, the site-stratified fold
construction, and the pathology-side slide embedding.
"""

from __future__ import annotations

from oaer.cohort.builder import (
    PROSPECTIVE_MARGINALS,
    RETROSPECTIVE_MARGINALS,
    CohortBuilder,
    build_cohort,
    exposure_prevalence,
    marginal_counts,
    marginal_table,
    modality_census,
)
from oaer.cohort.pathology import (
    TILE_GRID,
    SlideEmbedder,
    embed_slide,
    slide_signature,
)
from oaer.cohort.schema import (
    DECISION_COLUMNS,
    MOLECULAR_PANEL_COLUMNS,
    CohortSchemaError,
    RecordSchema,
    validate_record,
)
from oaer.cohort.splits import (
    SiteStratifiedFolds,
    cross_fitted_folds,
    fold_assignment,
    outer_site_split,
)

__all__ = [
    "DECISION_COLUMNS",
    "MOLECULAR_PANEL_COLUMNS",
    "PROSPECTIVE_MARGINALS",
    "RETROSPECTIVE_MARGINALS",
    "TILE_GRID",
    "CohortBuilder",
    "CohortSchemaError",
    "RecordSchema",
    "SiteStratifiedFolds",
    "SlideEmbedder",
    "cross_fitted_folds",
    "embed_slide",
    "exposure_prevalence",
    "fold_assignment",
    "build_cohort",
    "marginal_counts",
    "marginal_table",
    "modality_census",
    "outer_site_split",
    "slide_signature",
    "validate_record",
]
