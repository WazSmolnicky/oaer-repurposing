"""The schema-compatible cohort builder.

The private multi-centre arm is not redistributed, so the arm this package runs
on is built. The builder is schema-compatible with the declared contract
and holds every reported one-variable marginal exactly: the site cells of Table 1
panel a, and the molecular, line-of-therapy, sidedness, pattern, presentation,
performance-status, sex, age and modality-availability cells of Table 4, in both
the retrospective and the prospective arm. Only the marginals are matched; the
joint distribution of the arm it builds is its own and is not a claim
about the reported cohort.

The survival process is a proportional-hazards Weibull so that the quantity the
estimator has to recover is known in closed form: the hazard is
``k/lambda (t/lambda)^(k-1) exp(eta)`` with ``eta`` the risk score plus the
candidate's individual log-hazard effect, and the restricted mean of the
survival function is analytic. Censoring carries its own covariate
dependence, which is what makes the inverse-probability-of-censoring term of
Eq. (2) load-bearing rather than decorative.

Ref: Table 1 panel a, Table 4, Sec. 2.1, Sec. 2.3 (Eq. (2) needs G).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from oaer.support.seeding import spawn_generator
from oaer.support.types import (
    DAYS_PER_MONTH,
    GRACE_DAYS,
    LINE_HORIZON_MONTHS,
    ClinicalRecord,
    Decision,
    DecisionBatch,
    DecisionState,
    LineOfTherapy,
    MetastaticPattern,
    ModalityAvailability,
    MolecularPanel,
    MolecularStratum,
    OperatorStatus,
    PathologyProfile,
    Presentation,
    Sidedness,
    SlideSource,
    Split,
)

# Reported one-variable marginals. The right-hand side is the count the article
# prints, never a share this package chose.
RETROSPECTIVE_MARGINALS: Final[Mapping[str, Mapping[str, int]]] = {
    "site": {"A": 3412, "B": 2938, "C": 1612},
    "stratum": {
        "ras_mutant_mss": 3512,
        "ras_braf_wild_type_mss": 1902,
        "braf_v600e": 862,
        "msi_high": 486,
        "other_including_not_fully_typed": 1200,
    },
    "line": {"first": 2682, "second": 2914, "third_plus": 2366},
    "sidedness": {"right": 3186, "left": 4776},
    "pattern": {"liver_limited": 3104, "lung_only": 1286, "multi_organ": 3572},
    "presentation": {"synchronous": 4618, "metachronous": 3344},
    "ecog": {"zero_one": 6214, "two_or_above": 1748},
    "sex": {"female": 3586, "male": 4376},
    "age": {"under_65": 3514, "65_to_74": 2948, "75_or_older": 1500},
    "availability": {"all_three": 6958, "partial": 1004},
}

PROSPECTIVE_MARGINALS: Final[Mapping[str, Mapping[str, int]]] = {
    "site": {"A": 1187, "B": 983, "C": 544},
    "stratum": {
        "ras_mutant_mss": 1196,
        "ras_braf_wild_type_mss": 654,
        "braf_v600e": 298,
        "msi_high": 168,
        "other_including_not_fully_typed": 398,
    },
    "line": {"first": 918, "second": 996, "third_plus": 800},
    "sidedness": {"right": 1124, "left": 1590},
    "pattern": {"liver_limited": 1058, "lung_only": 438, "multi_organ": 1218},
    "presentation": {"synchronous": 1574, "metachronous": 1140},
    "ecog": {"zero_one": 2118, "two_or_above": 596},
    "sex": {"female": 1222, "male": 1492},
    "age": {"under_65": 1198, "65_to_74": 1004, "75_or_older": 512},
    "availability": {"all_three": 2372, "partial": 342},
}

# Baseline-hazard parameters. Declared engineering defaults: the article reports
# the estimated contrasts and the restricted means, not the hazard behind it.
BASELINE_SHAPE: Final[float] = 1.18
BASELINE_SCALE_MONTHS: Final[float] = 6.85
CENSORING_SHAPE: Final[float] = 1.05
CENSORING_SCALE_MONTHS: Final[float] = 17.5

# Candidate-level log-hazard effects. The four named agents carry the point
# estimate the article's emulation reports for that agent, so the recovery check
# has a target that is the article's own number rather than a number chosen here.
NAMED_EFFECTS: Final[Mapping[str, float]] = {
    "metformin": math.log(0.71),
    "bisoprolol": math.log(0.76),
    "aspirin": math.log(0.68),
    "omeprazole": math.log(1.18),
}

# The amplified-modifier gain of the pathology route: how much of a candidate's
# benefit varies with the slide-side modifier. Declared engineering default.
MODIFIER_GAIN: Final[Mapping[str, float]] = {
    "metformin": 0.22,
    "bisoprolol": 0.14,
    "aspirin": 0.31,
    "omeprazole": -0.10,
}
DEFAULT_EFFECT: Final[float] = math.log(0.77)
DEFAULT_MODIFIER_GAIN: Final[float] = 0.18

# Exposure prevalence per named agent, as Table 1 panel c reports it: the exposed
# count over the pooled decision count. These are the article's own numbers and
# the builder holds them, so the arm it builds carries the reported exposure
# marginals for the four named agents as well as the reported state marginals.
NAMED_EXPOSURE_PREVALENCE: Final[Mapping[str, float]] = {
    "metformin": 1286 / 7962,
    "bisoprolol": 962 / 7962,
    "aspirin": 534 / 7962,
    "omeprazole": 1704 / 7962,
}

# The exposure prevalence of a candidate the article does not name.
# Declared engineering default.
UNNAMED_EXPOSURE_PREVALENCE: Final[float] = 0.030

STRATUM_RISK: Final[Mapping[MolecularStratum, float]] = {
    MolecularStratum.RAS_MUTANT_MSS: 0.10,
    MolecularStratum.RAS_BRAF_WILD_TYPE_MSS: -0.14,
    MolecularStratum.BRAF_V600E: 0.32,
    MolecularStratum.MSI_HIGH: 0.04,
    MolecularStratum.OTHER: 0.16,
}

NAME_STRATUM: Final[Mapping[str, MolecularStratum]] = {
    "metformin": MolecularStratum.RAS_MUTANT_MSS,
    "bisoprolol": MolecularStratum.RAS_BRAF_WILD_TYPE_MSS,
    "aspirin": MolecularStratum.BRAF_V600E,
    "omeprazole": MolecularStratum.OTHER,
}

NAMED_LABELS: Final[tuple[str, ...]] = ("metformin", "bisoprolol", "aspirin", "omeprazole")


def interleave(counts: Mapping[str, int]) -> tuple[str, ...]:
    """Spread labelled slots so each label keeps its exact count.

    Each category's slots are placed at the midpoints of its equal shares of the
    line, and the union is sorted by slot. The result carries every category's
    reported count exactly and interleaves them rather than blocking them.
    """

    total = sum(counts.values())
    if total <= 0:
        raise ValueError("a marginal must carry at least one record")
    slots: list[tuple[float, int, str]] = []
    for order, (label, count) in enumerate(counts.items()):
        if count < 0:
            raise ValueError(f"the marginal for {label!r} is negative")
        for position in range(count):
            slots.append(((position + 0.5) * total / count, order, label))
    slots.sort()
    return tuple(label for _, _, label in slots)


def scaled_marginals(
    base: Mapping[str, Mapping[str, int]],
    total: int,
) -> dict[str, dict[str, int]]:
    """Rescale a reported marginal set to a target total, keeping its composition.

    The largest-remainder rule is used, so every cell is an integer, the total is
    exactly ``total`` and no cell vanishes unless its reported share is zero. A
    scaled arm keeps the reported composition and not the reported counts, which
    is why a scaled run's marginal match is reported against the scaled table.
    """

    if total < 1:
        raise ValueError("a scaled arm needs at least one record")
    scaled: dict[str, dict[str, int]] = {}
    for variable, counts in base.items():
        source_total = sum(counts.values())
        if source_total <= 0:
            raise ValueError(f"the marginal for {variable!r} is empty")
        exact = {label: count * total / source_total for label, count in counts.items()}
        floors = {label: int(value) for label, value in exact.items()}
        remainder = total - sum(floors.values())
        order = sorted(
            exact,
            key=lambda label: (-(exact[label] - floors[label]), label),
        )
        for position in range(remainder):
            floors[order[position % len(order)]] += 1
        scaled[variable] = floors
    return scaled


def marginal_table(counts: Mapping[str, int]) -> dict[str, int]:
    """A marginal as a plain dictionary, with the total attached."""

    table = dict(counts)
    table["_total"] = sum(counts.values())
    return table


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def calibrate_offset(logits: Sequence[float], rate: float) -> float:
    """The intercept that makes a tilted exposure model average to ``rate``.

    The confounding tilt is a function of the patient's state, so on its own it
    moves the marginal prevalence away from the rate the table reports. The offset
    is the scalar that restores the reported rate exactly in expectation, so the
    builder holds the reported marginals and still carries confounding; without
    it the tilt raises every one of the four named agents' prevalences.
    """

    if not logits:
        return 0.0
    target = min(max(rate, 1e-6), 1.0 - 1e-6)

    def mean_at(offset: float) -> float:
        return sum(sigmoid(value + offset) for value in logits) / len(logits)

    low, high = -60.0, 60.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if mean_at(middle) > target:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def exposure_pool(candidates: Sequence[str], named_only: bool) -> tuple[str, ...]:
    """The candidate pool an exposure draw runs over."""

    if named_only:
        return tuple(name for name in NAMED_LABELS if name in candidates) or NAMED_LABELS
    return tuple(candidates)


@dataclass
class CohortBuilder:
    """Generate a schema-compatible arm from the reported marginals."""

    seed: int = 20260831
    profile_dim: int = 16

    def build(
        self,
        marginals: Mapping[str, Mapping[str, int]],
        *,
        split_by_site: Mapping[str, Split],
        patients: int,
        candidates: Sequence[str] = (),
        named_only: bool = False,
    ) -> DecisionBatch:
        total = self._total(marginals)
        key = sorted(split_by_site.values())[0].value
        # The state is built before any exposure is drawn, because the exposure
        # rate is calibrated against the state the arm actually carries. The two
        # passes draw from their own streams, so the arm stays reproducible.
        state_stream = spawn_generator(self.seed, f"cohort-states:{key}")
        draw_stream = spawn_generator(self.seed, f"cohort-draws:{key}")
        assignment = {variable: interleave(counts) for variable, counts in marginals.items()}
        decisions_per_patient = total / patients
        candidates = tuple(candidates)
        pool = exposure_pool(candidates, named_only)
        built: list[tuple[DecisionState, float, ModalityAvailability, str, dict[str, float]]] = []
        tilts: dict[str, list[float]] = {name: [] for name in pool}
        for index in range(total):
            site = assignment["site"][index]
            line = LineOfTherapy(assignment["line"][index])
            family = assignment["stratum"][index]
            molecular = self._panel(family, state_stream)
            clinical = self._clinical(assignment, index, line, state_stream)
            availability = (
                ModalityAvailability.ALL_THREE
                if assignment["availability"][index] == "all_three"
                else ModalityAvailability.PARTIAL
            )
            pathology = self._pathology(split_by_site[site], molecular, state_stream)
            if availability is ModalityAvailability.PARTIAL:
                pathology = self._withhold_modality(pathology)
            state = DecisionState(molecular=molecular, pathology=pathology, clinical=clinical)
            prescribing = self._prescribing_set(state, state_stream)
            risk = decision_risk(state)
            for name, value in self._exposure_logits(state, risk, pool).items():
                tilts[name].append(value)
            built.append((state, risk, availability, site, prescribing))
        offsets = {
            name: calibrate_offset(
                values, NAMED_EXPOSURE_PREVALENCE.get(name, UNNAMED_EXPOSURE_PREVALENCE)
            )
            for name, values in tilts.items()
        }
        decisions: list[Decision] = []
        per_candidate_exposures: dict[str, int] = dict.fromkeys(candidates, 0)
        for index, (state, risk, availability, site, prescribing) in enumerate(built):
            exposure = {
                name: bool(draw_stream.random() < sigmoid(value + offsets[name]))
                for name, value in self._exposure_logits(state, risk, pool).items()
            }
            for name, flag in exposure.items():
                if flag:
                    per_candidate_exposures[name] = per_candidate_exposures.get(name, 0) + 1
            follow_up, event = self._follow_up(state, risk, exposure, draw_stream)
            patient = int(index // decisions_per_patient)
            decisions.append(
                Decision(
                    decision_id=f"{split_by_site[site].value}-{index:06d}",
                    patient_id=f"p{patient:06d}",
                    site=site,
                    split=split_by_site[site],
                    state=state,
                    prescribing_set=prescribing,
                    exposure=exposure,
                    follow_up_months=follow_up,
                    event=event,
                    backbone_regimen=("folfox", "folfiri", "folfoxiri", "capox")[index % 4],
                    objective_response=bool(
                        draw_stream.random() < 0.30 + 0.05 * (1.0 if risk < 0.0 else 0.0)
                    ),
                    modalities_available=availability,
                )
            )
        self.exposure_census = per_candidate_exposures
        return DecisionBatch(tuple(decisions))

    @staticmethod
    def _total(marginals: Mapping[str, Mapping[str, int]]) -> int:
        totals = {sum(counts.values()) for counts in marginals.values()}
        if len(totals) != 1:
            raise ValueError(f"the marginals disagree on the total: {sorted(totals)}")
        return totals.pop()

    @staticmethod
    def _panel(family: str, stream: np.random.Generator) -> MolecularPanel:
        if family == "msi_high":
            return MolecularPanel(
                ras_mutant=False,
                braf_v600e=False,
                msi_high=True,
                pi3k_altered=bool(stream.random() < 0.22),
                erbb2_amplified=bool(stream.random() < 0.06),
                kras_g12c=False,
                assay_panel=("narrow", "intermediate", "broad")[int(stream.integers(0, 3))],
                microsatellite_assay=("pcr", "ihc")[int(stream.integers(0, 2))],
            )
        if family == "braf_v600e":
            return MolecularPanel(
                ras_mutant=False,
                braf_v600e=True,
                msi_high=False,
                pi3k_altered=bool(stream.random() < 0.31),
                erbb2_amplified=False,
                kras_g12c=False,
                assay_panel=("narrow", "intermediate", "broad")[int(stream.integers(0, 3))],
                microsatellite_assay=("pcr", "ihc")[int(stream.integers(0, 2))],
            )
        if family == "ras_mutant_mss":
            return MolecularPanel(
                ras_mutant=True,
                braf_v600e=False,
                msi_high=False,
                pi3k_altered=bool(stream.random() < 0.19),
                erbb2_amplified=False,
                kras_g12c=bool(stream.random() < 0.05),
                assay_panel=("narrow", "intermediate", "broad")[int(stream.integers(0, 3))],
                microsatellite_assay=("pcr", "ihc")[int(stream.integers(0, 2))],
            )
        if family == "ras_braf_wild_type_mss":
            return MolecularPanel(
                ras_mutant=False,
                braf_v600e=False,
                msi_high=False,
                pi3k_altered=bool(stream.random() < 0.24),
                erbb2_amplified=False,
                kras_g12c=False,
                assay_panel=("narrow", "intermediate", "broad")[int(stream.integers(0, 3))],
                microsatellite_assay=("pcr", "ihc")[int(stream.integers(0, 2))],
            )
        # The residual block: a record not fully typed. One of the two extended
        # markers is left unassayed, which is exactly why the record is not
        # folded into the wild-type stratum.
        return MolecularPanel(
            ras_mutant=bool(stream.random() < 0.35),
            braf_v600e=False,
            msi_high=False,
            pi3k_altered=bool(stream.random() < 0.20),
            erbb2_amplified=None,
            kras_g12c=None if stream.random() < 0.5 else False,
            assay_panel="narrow",
            microsatellite_assay=("pcr", "ihc")[int(stream.integers(0, 2))],
        )

    @staticmethod
    def _withhold_modality(profile: PathologyProfile) -> PathologyProfile:
        """Drop the slide route from a record whose modality set is incomplete.

        A record with a missing modality loses its slide embedding, which is the
        only way the pathology route can be absent for a real cohort rather than
        for an ablation that withholds it.
        """

        return PathologyProfile(
            embedding=(),
            tile_count=0,
            slide_source=profile.slide_source,
            tumour_purity=profile.tumour_purity,
            morphological_features={},
        )

    @staticmethod
    def _clinical(
        assignment: Mapping[str, Sequence[str]],
        index: int,
        line: LineOfTherapy,
        stream: np.random.Generator,
    ) -> ClinicalRecord:
        age_band = assignment["age"][index]
        if age_band == "under_65":
            age = float(stream.uniform(38.0, 64.9))
        elif age_band == "65_to_74":
            age = float(stream.uniform(65.0, 74.9))
        else:
            age = float(stream.uniform(75.0, 92.0))
        return ClinicalRecord(
            age_years=age,
            female=assignment["sex"][index] == "female",
            ecog=(
                OperatorStatus.ZERO_ONE
                if assignment["ecog"][index] == "zero_one"
                else OperatorStatus.TWO_OR_ABOVE
            ),
            sidedness=(
                Sidedness.RIGHT if assignment["sidedness"][index] == "right" else Sidedness.LEFT
            ),
            pattern=MetastaticPattern(assignment["pattern"][index]),
            presentation=Presentation(assignment["presentation"][index]),
            line=line,
            prior_lines={"first": 0, "second": 1, "third_plus": int(stream.integers(2, 4))}[
                line.value
            ],
            organ_sites={
                MetastaticPattern.LIVER_LIMITED: 1,
                MetastaticPattern.LUNG_ONLY: 1,
                MetastaticPattern.MULTI_ORGAN: 3,
            }[MetastaticPattern(assignment["pattern"][index])],
            albumin_g_per_l=float(stream.normal(38.5, 4.5)),
            ldh_ratio_to_upper_limit=float(abs(stream.normal(1.05, 0.32)) + 0.35),
            cea_ng_per_ml=float(abs(stream.lognormal(1.55, 0.85))),
            comorbidity_count=int(stream.integers(0, 5)),
        )

    @staticmethod
    def _pathology(
        split: Split,
        molecular: MolecularPanel,
        stream: np.random.Generator,
    ) -> PathologyProfile:
        # Slide source is not independent of the site's acquisition route: a
        # held-out site contributes more biopsies than a development site does.
        weights = (0.60, 0.24, 0.16) if split is not Split.HELD_OUT_SITE_C else (0.42, 0.35, 0.23)
        source = SlideSource(
            str(stream.choice(("primary_resection", "primary_biopsy", "metastasis"), p=weights))
        )
        purity = float(np.clip(stream.normal(0.62, 0.13), 0.15, 0.95))
        embedding = tuple(float(value) for value in stream.normal(0.0, 1.0, size=16))
        return PathologyProfile(
            embedding=embedding,
            tile_count=int(stream.integers(3200, 24000)),
            slide_source=source,
            tumour_purity=purity,
            morphological_features={
                "stroma_ratio": float(np.clip(stream.normal(0.38, 0.11), 0.02, 0.9)),
                "mucin_ratio": float(np.clip(stream.normal(0.21, 0.09), 0.0, 0.8)),
                "tumour_budding": float(abs(stream.normal(1.4, 0.7))),
                "msi_morphology_score": float(
                    1.0 if molecular.msi_high else abs(stream.normal(0.3, 0.2))
                ),
            },
        )

    @staticmethod
    def _prescribing_set(state: DecisionState, stream: np.random.Generator) -> dict[str, float]:
        clinical = state.clinical
        molecular = state.molecular
        return {
            "age_years": (clinical.age_years - 66.0) / 10.0,
            "female": 1.0 if clinical.female else 0.0,
            "ecog_two_or_above": float(clinical.ecog.value),
            "right_sided": 1.0 if clinical.sidedness is Sidedness.RIGHT else 0.0,
            "line_index": float(clinical.prior_lines),
            "organ_sites": float(clinical.organ_sites),
            "albumin_scaled": (clinical.albumin_g_per_l - 38.5) / 5.0,
            "ldh_scaled": clinical.ldh_ratio_to_upper_limit - 1.4,
            "cea_log": math.log1p(clinical.cea_ng_per_ml) - 1.8,
            "comorbidities": float(clinical.comorbidity_count),
            "pi3k_altered": 1.0 if molecular.pi3k_altered else 0.0,
            "msi_high": 1.0 if molecular.msi_high else 0.0,
            "ras_mutant": 1.0 if molecular.ras_mutant else 0.0,
            "tumour_purity": state.pathology.tumour_purity - 0.62,
            "morphology_burden": state.pathology.morphological_features.get("tumour_budding", 0.0)
            - 1.4,
            "noise": float(stream.normal(0.0, 0.15)),
        }

    @staticmethod
    def _exposure_logits(
        state: DecisionState,
        risk: float,
        pool: Sequence[str],
    ) -> dict[str, float]:
        """The confounding tilt of each candidate's exposure, before calibration.

        Confounding: the propensity rises with the performance status and with the
        prior lines, both of which also raise the hazard. The tilt is returned
        without the intercept, which ``calibrate_offset`` solves for so the arm
        still carries the prevalences the table reports.
        """

        logits: dict[str, float] = {}
        for name in pool:
            logits[name] = (
                0.18 * float(state.clinical.ecog.value)
                + 0.05 * float(state.clinical.prior_lines)
                + 0.06 * (risk - 0.2)
            )
        return logits

    @staticmethod
    def _effect(name: str) -> tuple[float, float]:
        base = NAMED_EFFECTS.get(name, DEFAULT_EFFECT)
        gain = MODIFIER_GAIN.get(name, DEFAULT_MODIFIER_GAIN)
        return base, gain

    def _follow_up(
        self,
        state: DecisionState,
        risk: float,
        exposure: Mapping[str, bool],
        stream: np.random.Generator,
    ) -> tuple[float, bool]:
        horizon = state.line.horizon_months
        eta = risk
        modifier = decision_modifier(state)
        for name, flag in exposure.items():
            if not flag:
                continue
            base, gain = self._effect(name)
            eta += base + gain * modifier
        progression = self._weibull(
            eta,
            BASELINE_SHAPE,
            BASELINE_SCALE_MONTHS,
            stream,
        )
        administrative = horizon * 1.4
        censoring_eta = 0.30 * float(state.clinical.ecog.value) + 0.12 * float(
            state.clinical.prior_lines
        )
        if any(exposure.values()):
            censoring_eta += 0.10
        censoring = self._weibull(
            censoring_eta,
            CENSORING_SHAPE,
            CENSORING_SCALE_MONTHS,
            stream,
        )
        follow_up = min(progression, censoring, administrative)
        event = progression <= min(censoring, administrative)
        return max(follow_up, 0.02), bool(event)

    @staticmethod
    def _weibull(
        eta: float,
        shape: float,
        scale: float,
        stream: np.random.Generator,
    ) -> float:
        uniform = float(stream.random())
        uniform = min(max(uniform, 1e-9), 1.0 - 1e-9)
        draw: float = scale * ((-math.log(uniform)) / math.exp(eta)) ** (1.0 / shape)
        return draw


def build_cohort(
    *,
    seed: int = 20260831,
    candidates: Sequence[str] = (),
    named_only: bool = False,
    include_prospective: bool = True,
    total: int | None = None,
    patients: int | None = None,
) -> tuple[DecisionBatch, DecisionBatch]:
    """Generate the retrospective and, optionally, the prospective arm.

    The retrospective arm carries Sites A and B as development and Site C as the
    held-out site; the prospective arm keeps the same three sites, none of which
    the frozen model was fitted on. Passing ``total`` rescales both arms to that
    decision count while keeping the reported composition.
    """

    builder = CohortBuilder(seed=seed)
    retrospective_marginals: Mapping[str, Mapping[str, int]]
    prospective_marginals: Mapping[str, Mapping[str, int]]
    if total is None:
        retrospective_marginals = dict(RETROSPECTIVE_MARGINALS)
        prospective_marginals = dict(PROSPECTIVE_MARGINALS)
        retrospective_patients = 4318 if patients is None else patients
        prospective_patients = 1586 if patients is None else patients
    else:
        retrospective_total = sum(RETROSPECTIVE_MARGINALS["site"].values())
        prospective_total = sum(PROSPECTIVE_MARGINALS["site"].values())
        retrospective_marginals = scaled_marginals(RETROSPECTIVE_MARGINALS, total)
        prospective_marginals = scaled_marginals(
            PROSPECTIVE_MARGINALS, max(total * prospective_total // retrospective_total, 1)
        )
        retrospective_patients = max(int(total / 1.8439), 1)
        prospective_patients = max(int(retrospective_patients * 1586 // 4318), 1)
    retrospective = builder.build(
        retrospective_marginals,
        split_by_site={
            "A": Split.DEV_SITE_A,
            "B": Split.DEV_SITE_B,
            "C": Split.HELD_OUT_SITE_C,
        },
        patients=retrospective_patients,
        candidates=candidates,
        named_only=named_only,
    )
    if not include_prospective:
        return retrospective, DecisionBatch(())
    prospective = builder.build(
        prospective_marginals,
        split_by_site={
            "A": Split.PROSPECTIVE,
            "B": Split.PROSPECTIVE,
            "C": Split.PROSPECTIVE,
        },
        patients=prospective_patients,
        candidates=candidates,
        named_only=named_only,
    )
    return retrospective, prospective


def decision_risk(state: DecisionState) -> float:
    """The builder's log-hazard for a state, before any exposure.

    Published rather than private because the arm's own average effect is a
    function of it, and a check that reads the arm's truth should not have to
    reach into the builder to find it.
    """

    clinical = state.clinical
    risk = STRATUM_RISK[state.molecular.stratum]
    risk += 0.42 * float(clinical.ecog.value)
    risk += 0.09 * (clinical.age_years - 66.0) / 10.0
    risk += 0.16 * float(clinical.prior_lines)
    risk += 0.13 * (clinical.ldh_ratio_to_upper_limit - 1.4)
    risk -= 0.11 * (clinical.albumin_g_per_l - 38.5) / 5.0
    risk += 0.05 * math.log1p(clinical.cea_ng_per_ml)
    risk -= 0.08 if clinical.sidedness is Sidedness.LEFT else 0.0
    risk += 0.11 * float(clinical.organ_sites)
    return risk


def decision_modifier(state: DecisionState) -> float:
    """The slide-side benefit modifier of a state, centred at the arm's mean."""

    features = state.pathology.morphological_features
    if not features:
        return 0.0
    return 0.6 * (features["stroma_ratio"] - 0.38) / 0.11 + 0.4 * (features["tumour_budding"] - 1.4)


def reported_restricted_mean(
    eta: float, horizon: float, shape: float = BASELINE_SHAPE, scale: float = BASELINE_SCALE_MONTHS
) -> float:
    """The builder's closed-form restricted mean of the Weibull survival.

    ``E[min(T, t*)] = int_0^{t*} exp(-(t/scale)^shape exp(eta)) dt``, evaluated by
    the trapezoid rule on a fine grid. The check compares this against the Monte
    Carlo mean of the arm it builds rather than against the estimator.
    """

    grid = np.linspace(0.0, horizon, 4001)
    survival = np.exp(-((grid / scale) ** shape) * math.exp(eta))
    return float(np.trapezoid(survival, grid))


def reported_hazard_ratio(effect: float) -> float:
    """The builder's closed-form hazard ratio for a log-hazard effect."""

    return math.exp(effect)


def grace_days(line: LineOfTherapy) -> int:
    return GRACE_DAYS[line.value]


def horizon_months(line: LineOfTherapy) -> float:
    return LINE_HORIZON_MONTHS[line.value]


def days_to_months(days: float) -> float:
    return days / DAYS_PER_MONTH


def exposure_prevalence(batch: DecisionBatch, candidate: str) -> float:
    """The share of decisions exposed to one candidate."""

    if len(batch) == 0:
        return 0.0
    flags: npt.NDArray[np.float64] = np.asarray(
        [1.0 if decision.exposure.get(candidate, False) else 0.0 for decision in batch.decisions],
        dtype=np.float64,
    )
    return float(flags.mean())


def modality_census(batch: DecisionBatch) -> dict[str, int]:
    """Availability counts over an arm.

    A decision carries all three modalities when its slide, its molecular panel
    and its structured timeline are all recorded; the comparison is run against
    the withheld-modality reference rather than against the full-modality one.
    """

    counts = {"all_three": 0, "at_least_one_missing": 0}
    for decision in batch.decisions:
        if decision.modalities_available is ModalityAvailability.ALL_THREE:
            counts["all_three"] += 1
        else:
            counts["at_least_one_missing"] += 1
    return counts


def marginal_counts(batch: DecisionBatch, variable: str) -> dict[str, int]:
    """The realised one-variable marginal of an arm this release builds."""

    counts: dict[str, int] = {}
    for decision in batch.decisions:
        key = _marginal_key(decision, variable)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _marginal_key(decision: Decision, variable: str) -> str:
    clinical = decision.state.clinical
    if variable == "site":
        return decision.site
    if variable == "stratum":
        return decision.state.molecular.stratum.value
    if variable == "line":
        return clinical.line.value
    if variable == "sidedness":
        return "right" if clinical.sidedness is Sidedness.RIGHT else "left"
    if variable == "pattern":
        return clinical.pattern.value
    if variable == "presentation":
        return clinical.presentation.value
    if variable == "ecog":
        return "zero_one" if clinical.ecog is OperatorStatus.ZERO_ONE else "two_or_above"
    if variable == "sex":
        return "female" if clinical.female else "male"
    if variable == "age":
        if clinical.age_years < 65.0:
            return "under_65"
        if clinical.age_years < 75.0:
            return "65_to_74"
        return "75_or_older"
    if variable == "availability":
        return (
            "all_three"
            if decision.modalities_available is ModalityAvailability.ALL_THREE
            else "partial"
        )
    raise KeyError(f"{variable!r} is not a reported marginal")
