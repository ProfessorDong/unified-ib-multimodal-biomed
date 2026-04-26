# Unified Information Bottleneck Framework for Multimodal Biomedical ML

Code and aggregate results accompanying:

> Dong, L. **A Unified Information Bottleneck Framework for Multimodal Biomedical Machine Learning.** *Entropy* **2026**, *28*(4), 445. [https://doi.org/10.3390/e28040445](https://doi.org/10.3390/e28040445)

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fe28040445-blue)](https://doi.org/10.3390/e28040445)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- Add Zenodo DOI badge after first GitHub release is archived to Zenodo. -->

The paper develops an information-theoretic framework for multimodal biomedical machine learning, formulating multimodal representation learning through the information bottleneck principle and providing tools for: mutual-information decomposition across modalities, redundancy/synergy quantification, fusion-collapse diagnostics, missing-modality robustness as information consistency, longitudinal modeling via transfer entropy, and uncertainty / calibration / OOD detection. Empirical case studies span three datasets (TCGA-BRCA, TCGA-GBMLGG, OASIS-2).

This repository contains:

- `experiments/` — Python source for all empirical case studies in Section 9
- `experiments/results/` — aggregate result JSON files reported in the paper
- `environment.yml`, `requirements.txt` — reproducible environment specs
- `CITATION.cff` — machine-readable citation

This repository **does not** redistribute any patient-derived data.

---

## Quick start

```bash
git clone https://github.com/ProfessorDong/unified-ib-multimodal-biomed.git
cd unified-ib-multimodal-biomed
conda env create -f environment.yml
conda activate nih_research
# obtain data per the Data Availability section below, then:
cd experiments && python3 run_all.py
```

See [`experiments/README.md`](experiments/README.md) for per-script details and expected paths.

---

## Data Availability

Reproduced verbatim from the paper:

> The empirical studies in Section 9 use three publicly available datasets:
> (1) the TCGA-BRCA multi-omics dataset as preprocessed by Wang et al. (available at <https://github.com/txWang/MOGONET>, accessed on 15 February 2026);
> (2) the TCGA-GBMLGG clinical and genomic dataset from Chen et al. (available at <https://github.com/mahmoodlab/MCAT>, accessed on 15 February 2026); and
> (3) the OASIS-2 longitudinal Alzheimer's MRI dataset (available at <https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers>, accessed on 15 February 2026).
> No new primary data were generated.

OASIS-2 is governed by a Data Use Agreement; users must register and agree to OASIS terms before downloading. After download, place files so the paths in `experiments/README.md` resolve.

---

## Selected results

| Quantity | Value |
| --- | --- |
| mRNA mutual information, BRCA | 0.878 ± 0.044 nats (64.9% of H(Y)) |
| Synergy S, BRCA same-level fusion | ≈ −0.63 (negative across 5/5 folds) |
| Synergy S, GBMLGG cross-level | ≈ −0.16 (near zero) |
| Best unimodal AUC, BRCA | 0.928 ± 0.013 |
| VMIB AUC, BRCA | 0.915 ± 0.016 |
| Missing-modality worst case (Standard / Dropout) | 0.668 / 0.796 |
| Fusion balance index (Standard / Dropout) | 0.25 / 0.62 |
| OOD entropy ratio | 2.0× |
| Selective prediction at 50% coverage | 0.939 accuracy (vs 0.787 base) |
| Sequential prediction gain, OASIS-2 | +0.022 AUC |
| Consistency-model degradation reduction | 37% |

Full numbers, including confidence intervals, are in `experiments/results/*.json` and the paper.

---

## How to cite

Preferred (paper):

```bibtex
@article{e28040445,
  author  = {Dong, Liang},
  title   = {A Unified Information Bottleneck Framework for Multimodal Biomedical Machine Learning},
  journal = {Entropy},
  volume  = {28},
  year    = {2026},
  number  = {4},
  pages   = {445},
  doi     = {10.3390/e28040445},
  url     = {https://www.mdpi.com/1099-4300/28/4/445},
  issn    = {1099-4300}
}
```

Software release: see `CITATION.cff` (a Zenodo DOI will be added after the first GitHub release is archived to Zenodo).

---

## Funding

This research was funded by the National Cancer Institute (NCI) of the National Institutes of Health (NIH) under Grant **R01 CA309499**.

## License

Code is released under the [MIT License](LICENSE). The published paper is open access under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) at the MDPI Entropy DOI above.

## Contact

Liang Dong — Baylor University / UT Southwestern Medical Center.
For issues with the code, please open a GitHub issue.
