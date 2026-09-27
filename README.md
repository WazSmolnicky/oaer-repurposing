# Outcome-anchored effect ranking over a typed knowledge graph

This package implements the model, the estimand and the read-outs reported in
*A graph foundation model linking molecular, pathology and real-world clinical
data for repurposable therapeutics in metastatic colorectal cancer*. Its central
object is the outcome-anchored effect ranking: a patient's molecular panel, the
slide embedding of their resection or biopsy, and their structured treatment
history are attached to a typed biomedical knowledge graph through the patient
node; a relational encoder pretrained on that graph by masked-edge and
anchor-reconstruction objectives produces a patient-candidate representation; a
shared ranking head turns that representation into a per-candidate
restricted-mean survival difference; and an identification gate pairs the ranking
with a weighted split-conformal interval so the model can abstain. The estimand is
a patient-level conditional effect estimated by a doubly robust pseudo-outcome
inside a clone-censor-weight target-trial emulation, not a prognostic score.

The private multi-centre arm of the article is not redistributed. What ships is the
record schema that arm declares, a schema-compatible builder whose output holds
the reported one-variable marginals cell for cell, every estimator and read-out the
article defines, and the article's own reported tables as data so the two can be
read side by side. Every cohort-level column of Tables 1-5 is therefore reported as
`NOT_RUN` against the article's arm, while the mechanisms are executed on the arm
this release builds; `verification_report.json` lists exactly which is which.

## Installation

Python 3.11 or newer.

    python3 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .          # optional: exposes the oaer-* entry points

Conda:

    conda env create -f environment.yml
    conda activate oaer

Container:

    docker build -t oaer .

The container image is pinned to `python:3.12-slim` with the CPU wheels of the
pinned dependency set, and its entry point is the full protocol. No image was built
while producing this release, because the machine that produced it carries no
`docker` binary, so `docker build` is unverified here and reported as `BLOCKED`.

## Data

Nothing in this package redistributes a third-party record. Each source below is
read at its own address under its own terms; `NOTICE` restates the licence and role
of every one of them, and `dataset_urls.txt` holds the access routes that were
fetched and read back while the release was assembled.

| resource | version read back | licence | role |
|---|---|---|---|
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | ChEMBL_37 (2026-05-01), from the status endpoint | CC BY-SA 3.0 | approved indications and mechanism of action |
| [Open Targets](https://platform.opentargets.org/) | live GraphQL `meta` field | CC0 1.0 | target-disease association, pathway layer |
| [DrugCentral](https://drugcentral.org/) | current release | CC BY-SA 4.0 | drug-target activity, active-comparator catalogue |
| [GDC TCGA-COADREAD](https://api.gdc.cancer.gov/projects/TCGA-COAD) | open tier | NIH GDS, open tier | adjacent-domain molecular and pathology cohort |
| [CPTAC-COAD](https://www.cbioportal.org/) (`coad_cptac_2019`) | open tier | NIH terms | adjacent-domain proteogenomic cohort, no follow-up fields |
| [MSK-IMPACT CRC](https://www.cbioportal.org/) (`crc_msk_2017`) | public study table | study's own terms | metastatic clinicogenomic cohort |
| [MSK-CHORD](https://www.cbioportal.org/) (`msk_chord_2024`) | public study table | study's own terms | metastatic real-world cohort |
| [SurGen](https://zenodo.org/records/14047723) | deposit as published | CC BY 4.0 | pathology-only cohort, patch embeddings |
| [GSE39582](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39582) | series matrix | NCBI GEO terms | transcriptomic adjacent-domain cohort |
| [TCGA-CRC-DX](https://zenodo.org/records/2530835) | deposit as published | CC BY 4.0 | microsatellite classification sanity probe |

Two resources are deliberately absent. The multi-centre arm of Sites A-C is private
clinical data that was irreversibly anonymised at the sites that generated it, and
no patient row exists in this tree. PAIP2020 is never used as an evidence source,
because its download sits behind a click-through research-use gate.

Nothing has to be downloaded to run the release. The auxiliary resources are read
only by the live checks:

    pipelines/verify_release.sh            # includes the live resource probes
    pipelines/verify_release.sh --no-live  # offline; the live entries become absent

Each probe is retried three times, and one that still cannot be read is reported
`BLOCKED` with its reason rather than omitted: an unreachable resource and an
absent row are different statements. Any `BLOCKED` probe leaves the release-level
verdict at `PARTIALLY_VERIFIED`, which is the honest reading — the tree is not at
fault, and neither is the resource. Transport from the machine that assembled it is
not uniform: the GSE39582 series matrix sits on an NCBI FTP host whose TLS
handshake this Python build abandons with an unexpected EOF, so the system HTTP
client was used to read that one URL, and that is the read-back `dataset_urls.txt`
records for it.

The cohort the estimators run on is built from the reported marginals:

    python3 -m oaer.console.rank --protocol protocols/_smoke.yaml --print

`protocols/_smoke.yaml` is for smoke use only and is never a reported value. The
full-scale protocol is `protocols/main.yaml`.

### On-disk footprint

The arm is built in memory from a seed, so its footprint is a few
megabytes and there is no dataset root to populate. The release tree itself is
about 1.5 MB. `integrity_manifest.json` records a SHA-256 per tracked file in path
order and excludes only itself; re-derive the whole set with

    python3 -m oaer.console.verify --protocol protocols/main.yaml

which rewrites the manifest last and then re-hashes the live tree to confirm the
digests still match.

## Running the protocol

The pipeline scripts each set `PYTHONPATH` to `src/` and are runnable from the
repository root.

| command | what it does | report |
|---|---|---|
| `pipelines/stage_graph.sh protocols/main.yaml` | Stage 1 and Stage 2 on the typed graph | `reports/pretrain_main.txt` |
| `pipelines/emulate_candidates.sh protocols/main.yaml` | cross-fitted Eq. (2) table per candidate | `reports/emulate_main.txt` |
| `pipelines/run_protocol.sh protocols/main.yaml` | pretraining, emulation, head, gate, read-outs | `reports/study_main.txt` |
| `pipelines/ablate_components.sh protocols/main.yaml` | the four canonical removals, their joints, and the removal of the candidate-balanced weights | `reports/ablate_main.txt` |
| `pipelines/render_ledger.sh protocols/main.yaml` | the article's reported tables, side by side | `reports/ledger_main.txt` |
| `pipelines/verify_release.sh protocols/main.yaml` | the two verification passes | `verification_report.json`, `reports/verification.txt`, `integrity_manifest.json` |

Every protocol is a plain YAML file under `protocols/` and every key can be
overridden on the command line, so a run needs no edit:

    python3 -m oaer.console.rank --protocol protocols/main.yaml \
        --set fitting.stage3_epochs=80 --set emulation.folds=5

Reported configurations:

- `protocols/main.yaml` — the deployed protocol
- `protocols/ablation_components.yaml` — Sec. 1.4 panel a
- `protocols/ablation_kg_terciles.yaml` — Sec. 1.4 panel b
- `protocols/ablation_modalities.yaml` — Sec. 1.4 panel c
- `protocols/sensitivity_site_holdout.yaml` — Table 5 panels a-b
- `protocols/supplementary_seed_panel.yaml` — the ten fixed seeds of Table 2
- `protocols/transfer_public_cohorts.yaml` — Table 5 panel d

## What a run reports, and what to compare it against

`reports/study_main.txt` carries the run's own read-outs and, at its foot, the
article's reported criteria for comparison. The article's cohort-level values are
not reproducible from this tree, so the comparison is directional and the article's
values are quoted with the interval it prints:

| read-out | article's reported value | this release |
|---|---|---|
| prospective PFS hazard ratio, clone-censor-weight | 0.68 (95% CI 0.55-0.84) | on the arm this release builds; not comparable |
| weighted restricted-mean policy contrast, k = 3 | 2.4 months (0.9-3.9) | on the arm this release builds; not comparable |
| prospective OS C-index, OAER against the reproduced reference | 0.746 against 0.658 | on the arm this release builds; not comparable |
| ranking, Hits@20 / nDCG@20 | 41.7 / 0.612 | on the arm this release builds; not comparable |
| prospective deferral rate | 28.4% | on the arm this release builds; not comparable |

Three things about the run *are* comparable, and `reports/verification.txt`
reports them as such:

- the **closed-form checks**. The hazard behind the arm is a Weibull proportional-
  hazards model, so its restricted mean has a closed form; Eq. (2) under oracle
  nuisances must collapse to the difference of the two arm means; a motif operator
  must equal an enumerated walk; a hazard ratio must equal a grid search on the
  written partial likelihood. Each is recomputed by hand or by brute force and
  required to agree.
- the **article's own table arithmetic**. Each Table 3 panel-d interaction is
  re-derived from the three panel-a rows its label names, and the cohort, subgroup
  and criteria ledgers are checked against their own pooled rows.
- the **executed training path**. A forward pass, the Eq. (3) loss, a backward pass,
  an optimiser step, a single-batch overfit, a minimal training loop and a
  checkpoint round trip all run and report what they saw.

The one ledger finding worth reading before the tables: the article reads its
pre-specified discrimination margin of +0.05 against the published spread of
reproduced prognostic scores. On the rows the article itself reports, that family
spans 0.618 to 0.658, a width of 0.040. A margin wider than the whole family cannot
be realised by any member of it, so the sensitivity row's not-met verdict follows
from the arithmetic rather than from the cohort. The check
`discrimination_margin_gap` reports both cutoffs and is filed in the `manuscript`
family, so it is not read as a defect in this tree.

## Compute

The article reports no accelerator, no memory footprint and no training hardware.
What it does report is the deployed model's inference cost: about **11 minutes per
qualifying decision, interquartile range 7-18 minutes**, with the ranking frozen for
**96.4%** of qualifying decisions. Those three figures are carried verbatim in
`protocols/main.yaml` under `compute`, and no other hardware claim is made anywhere
in this tree.

The whole release therefore runs on CPU. `protocols/main.yaml` scores a directory
of 24 candidates, which is the article's shape rather than its size (the article
never prints `|D| = K`); at that width the full protocol, including the ablation
sweep, completes in minutes on one core, and the reported 11-minute inference
figure is an order of magnitude above anything a run of this tree measures. Disk
footprint is the release tree, about 1.5 MB. Precision is fp32 and the device is
`cpu` by default; both are configuration keys (`fitting.precision`, `fitting.device`).

Every hyperparameter the article does not print is listed as a declared engineering
default in the `deviations` block of `claim_to_code.json`, with the paper location
it departs from and the reason it was chosen, so a reader can tell a reported value
from a chosen one without reading the source.

## Layout

    protocols/        one YAML per reported configuration, plus the smoke protocol
    catalogue/        the admission rule, the named agents and the auxiliary sources
    pipelines/        one shell entry point per command
    src/oaer/
      catalogue/      candidate admission, the typed graph, the motif algebra
      cohort/         record schema, schema-compatible builder, folds, slide route
      encoder/        motif propagation, the relational encoder, the anchor head
      estimands/      horizons, nuisances, Eq. (2), trimming, sensitivity, cloning
      head/           the route score, the centred modifier, Eq. (5), the ranker
      gate/           the eligible set, the conformal rule, the four gate outputs
      policy/         the Cox hazard ratio, the restricted mean, the top-k value
      readings/       ranking, discrimination, interference, resampling
      fitting/        optimiser and schedule, the loop, the three stages, checkpoints
      ledger/         the article's Tables 1-5 and the pre-specified criteria, as data
      validation/     the verification battery, the live probes, the manifest
      console/        one command per entry point
    tests/            unit, integration and end-to-end coverage
    reports/          plain-text reports written by the commands above

## Tests

    PYTHONPATH=src python3 -m pytest -q

The suite covers the catalogue and admission rule, the cohort schema and builder,
the encoder and its two pretraining stages, the estimand and its nuisances, the
gate and the conformal rule, the ranking head, the readings, the ledgers, and a
two-step closed training loop over the smoke protocol. Static checks:

    ruff check .
    black --check .
    isort --check-only .
    mypy            # strict over src/oaer

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`; the latter restates the licence and access
route of every third-party resource the package reads.
