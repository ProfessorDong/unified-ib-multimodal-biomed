# Experiments

Python source for the empirical case studies in Section 9 of the paper, plus the aggregate result JSON files committed under `results/`.

## Environment

Either route works; both reproduce the versions used to generate the committed results.

Conda (recommended, includes CUDA 12.6 build of PyTorch):
```bash
conda env create -f ../environment.yml
conda activate nih_research
```

Pip:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
```

## Data

This repository does **not** redistribute data. The three datasets are obtained directly from their original authors, exactly as in the paper's Data Availability Statement:

1. **TCGA-BRCA multi-omics** (preprocessed by Wang et al.): https://github.com/txWang/MOGONET
2. **TCGA-GBMLGG clinical + genomic** (Chen et al.): https://github.com/mahmoodlab/MCAT
3. **OASIS-2 longitudinal Alzheimer's MRI**: https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers

After downloading, place the data so the paths used by the scripts resolve:

```
experiments/
  data/
    MOGONET/BRCA/...           # from txWang/MOGONET
    MCAT/dataset_csv/...       # from mahmoodlab/MCAT
    OASIS2/oasis_longitudinal.csv
```

OASIS-2 is governed by a Data Use Agreement. Users must register and agree to OASIS terms before downloading.

## Running

All experiments:
```bash
python3 run_all.py
```

Individual experiments (matching the script names referenced in the paper):
```bash
python3 exp0_synthetic_synergy.py     # synthetic synergy model
python3 exp1_consistent_mi.py         # 5-fold CV MI estimates with bootstrap CIs (BRCA + GBMLGG)
python3 exp2_vmib_tradeoff.py         # VMIB compression-prediction tradeoff
python3 exp3_missing_modality_v2.py   # missing-modality robustness
python3 exp4_fusion_collapse_v2.py    # fusion-collapse diagnostics
python3 exp_baselines.py              # 5-fold CV baselines
python3 exp_uncertainty.py            # OOD detection, calibration, selective prediction
python3 exp_longitudinal.py           # OASIS-2 transfer entropy and sequential prediction
python3 exp_foundation.py             # representation entropy, adaptation efficiency
python3 exp_gbmlgg_synergy.py         # cross-level modality contrast
```

Each script writes a JSON file under `results/`. The committed JSONs are the runs reported in the paper; rerunning will overwrite them.

## Hardware

The experiments were run on a single CUDA-capable GPU. CPU-only execution works for everything except the largest sweeps and will take noticeably longer.

## Reproducibility notes

Random seeds are set inside the scripts where applicable. Small numerical differences between runs are expected from CUDA non-determinism and 5-fold CV shuffles; aggregate values (means, std, AUC) should match the paper to within the reported confidence intervals.
