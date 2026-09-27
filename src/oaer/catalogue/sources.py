"""Delineation of the auxiliary public resources the catalogue and the checks read.

The manuscript states that no resource used here needs a click-through agreement
and that each access route was tested for downloadability on the day of access.
That is what this module records: for each resource, the version read back, the
licence, the access route, the role it plays, and what a check should see. A
resource that could not be read back is carried as blocked with its reason rather
than omitted.

Ref: Sec. 2.2 (public auxiliary data and knowledge-graph construction), Table 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class AuxiliarySource:
    """One third-party resource with its version, licence and access route."""

    name: str
    version: str
    licence: str
    url: str
    role: str
    probe: str
    expect: str
    live: bool = True


AUXILIARY_SOURCES: Final[tuple[AuxiliarySource, ...]] = (
    AuxiliarySource(
        name="ChEMBL",
        version="ChEMBL_37 (release date 2026-05-01)",
        licence="CC BY-SA 3.0",
        url="https://www.ebi.ac.uk/chembl/api/data/status.json",
        role="approved indications and mechanism of action for the candidate catalogue",
        probe="status",
        expect="chembl_release_date",
    ),
    AuxiliarySource(
        name="Open Targets Platform",
        version="data release read back from the live GraphQL meta field",
        licence="CC0 1.0",
        url="https://api.platform.opentargets.org/api/v4/graphql",
        role="target-disease association scores and the pathway layer of the graph",
        probe="graphql_meta",
        expect="meta",
    ),
    AuxiliarySource(
        name="DrugCentral",
        version="current release at the access date",
        licence="CC BY-SA 4.0",
        url="https://drugcentral.org/download",
        role="drug-target activity and the active-comparator indication catalogue",
        probe="download_page",
        expect="download",
    ),
    AuxiliarySource(
        name="GDC projects API",
        version="live at the access date",
        licence="NIH data-use terms; open tier only",
        url="https://api.gdc.cancer.gov/projects/TCGA-COAD",
        role="open-tier record of the adjacent-domain colorectal cohorts",
        probe="gdc_project",
        expect="TCGA-COAD",
    ),
    AuxiliarySource(
        name="cBioPortal studies API",
        version="live at the access date",
        licence="study-specific; each study keeps its own terms",
        url="https://www.cbioportal.org/api/studies",
        role="clinicogenomic record for the MSK-IMPACT and MSK-CHORD strata",
        probe="cbioportal_studies",
        expect="msk_chord_2024",
    ),
    AuxiliarySource(
        name="GEO series GSE39582",
        version="series matrix, release current at the access date",
        licence="NCBI GEO terms of use",
        url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE39nnn/GSE39582/matrix/GSE39582_series_matrix.txt.gz",
        role="transcriptomic adjacent-domain cohort",
        probe="geo_matrix",
        expect="GSE39582",
    ),
    AuxiliarySource(
        name="TCGA-CRC-DX",
        version="Zenodo deposit 2530835, CC BY 4.0",
        licence="CC BY 4.0",
        url="https://zenodo.org/api/records/2530835",
        role="microsatellite classification sanity probe",
        probe="zenodo_record",
        expect="CRC_DX",
    ),
    AuxiliarySource(
        name="SurGen",
        version="Zenodo deposit 14047723, CC BY 4.0",
        licence="CC BY 4.0",
        url="https://zenodo.org/api/records/14047723",
        role="pathology-only adjacent-domain cohort",
        probe="zenodo_record",
        expect="SurGen",
    ),
)

# Sources that supply candidates rather than cohorts. The catalogue is the
# intersection of these three; an agent enters only when its target label carries
# a non-oncological approved indication.
CANDIDATE_SOURCES: Final[tuple[str, ...]] = ("ChEMBL", "Open Targets Platform", "DrugCentral")


@dataclass(frozen=True)
class PublicCohort:
    """One independent public cohort, as Table 5 panel d lists it."""

    name: str
    domain: str
    modalities: tuple[str, ...]
    records: int
    c_index: float | None
    delta_vs_matched_reference: float | None
    access: str
    redistributed: bool = False


PUBLIC_COHORTS: Final[tuple[PublicCohort, ...]] = (
    PublicCohort(
        name="TCGA-COADREAD",
        domain="adjacent, non-metastatic",
        modalities=("molecular", "pathology"),
        records=296,
        c_index=0.681,
        delta_vs_matched_reference=-0.021,
        access="GDC open tier",
    ),
    PublicCohort(
        name="CPTAC-COAD",
        domain="adjacent, proteogenomic",
        modalities=("molecular",),
        records=110,
        c_index=None,
        delta_vs_matched_reference=None,
        access="GDC open tier; the release carries no follow-up fields",
    ),
    PublicCohort(
        name="MSK-IMPACT colorectal",
        domain="metastatic, clinicogenomic",
        modalities=("molecular", "structured record"),
        records=1134,
        c_index=0.668,
        delta_vs_matched_reference=-0.027,
        access="cBioPortal study crc_msk_2017",
    ),
    PublicCohort(
        name="MSK-CHORD colorectal",
        domain="metastatic, real-world",
        modalities=("molecular", "structured record"),
        records=2316,
        c_index=0.672,
        delta_vs_matched_reference=-0.024,
        access="cBioPortal study msk_chord_2024",
    ),
    PublicCohort(
        name="SurGen, 426 survival cases",
        domain="adjacent, predominantly non-metastatic",
        modalities=("pathology",),
        records=426,
        c_index=0.681,
        delta_vs_matched_reference=-0.031,
        access="Zenodo deposit 14047723",
    ),
    PublicCohort(
        name="SurGen, metastatic subset of the 426",
        domain="metastatic, pathology only",
        modalities=("pathology",),
        records=168,
        c_index=0.669,
        delta_vs_matched_reference=-0.034,
        access="Zenodo deposit 14047723",
    ),
    PublicCohort(
        name="GSE39582",
        domain="adjacent, transcriptomic",
        modalities=("molecular",),
        records=585,
        c_index=0.652,
        delta_vs_matched_reference=-0.038,
        access="GEO series matrix over FTP",
    ),
    PublicCohort(
        name="Microsatellite probe, TCGA-CRC-DX",
        domain="sanity probe only; label AUROC 0.843 against chance",
        modalities=("pathology",),
        records=4112,
        c_index=None,
        delta_vs_matched_reference=None,
        access="Zenodo deposit 2530835",
    ),
)


def candidate_source_envelope() -> dict[str, tuple[str, str]]:
    """The version and licence of each catalogue source, keyed by name."""

    chosen = {source.name for source in AUXILIARY_SOURCES} & set(CANDIDATE_SOURCES)
    return {
        source.name: (source.version, source.licence)
        for source in AUXILIARY_SOURCES
        if source.name in chosen
    }
