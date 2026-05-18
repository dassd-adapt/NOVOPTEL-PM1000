# NOVOPTEL-PM1000

## Overview

This repository contains data and analysis tools for the NOVOPTEL PM1000 optical
polarimeter dataset (dataset-1603), which records Stokes-parameter trajectories on the
Poincaré sphere for five optical source types under five cable-disturbance events.

## Repository Structure

```
NOVOPTEL-PM1000/
├── dataset-1603/               ← 65+ CSV measurement files
├── analysis/
│   ├── feature_extraction.py   ← Windowed feature extraction (25 features)
│   └── features_windowed.csv   ← Generated output (gitignored)
├── ml_pipeline_universal.py    ← Universal ML pipeline (this document)
├── README.md
└── results.zip                 ← Pre-computed result archive
```

---

## Universal ML Pipeline (`ml_pipeline_universal.py`)

A production-ready, modular machine learning pipeline that processes **one source
type at a time** and produces a complete set of outputs including trained models,
evaluation metrics, visualisations and reproducibility artefacts.

### Quick start

```bash
# Install dependencies (once)
pip install numpy pandas scipy scikit-learn xgboost matplotlib seaborn pyyaml

# Run on 10GE source
python ml_pipeline_universal.py \
    --data_dir dataset-1603 \
    --source   10GE \
    --output_dir outputs_10GE_enhanced

# Run on DPQAM16 source
python ml_pipeline_universal.py \
    --data_dir dataset-1603 \
    --source   DPQAM16 \
    --output_dir outputs_DPQAM16_enhanced
```

### Supported sources

| `--source` value | Filename token    | Description                    |
|-----------------|-------------------|--------------------------------|
| `10GE`          | `10GE`            | 10 Gigabit Ethernet            |
| `DPQAM16`       | `DPQAM16-200G`    | Dual-polarisation 16-QAM 200 G |
| `DPQPSK`        | `DPQPSK-200G`     | Dual-polarisation QPSK 200 G   |
| `SP-PURE`       | `SP-PURE`         | Single-polarisation pure laser  |
| `SP-AGIL`       | `SP-AGIL`         | Single-polarisation agile laser |

### Event classes

| Code  | Name                    |
|-------|-------------------------|
| `NE`  | No event (baseline)     |
| `FS`  | Fiber squeeze           |
| `MB`  | Modal birefringence     |
| `TAP` | Fiber tap               |
| `VB`  | Vibration / bend        |

### Input data format

- **Location**: `dataset-1603/`
- **Filename pattern**: `pm1000_sop_2kHz_1min_{SOURCE}_{EVENT}_1550_{DATE}_{N}.csv`
- **Columns**: `Time_s, S0, S1, S2, S3`  
  `S1, S2, S3` are pre-normalised to the unit Poincaré sphere by the instrument
- **Duration**: ~60 seconds per file; ~120 000 rows

### Feature extraction (1 035 features per 1-second window)

**Time-domain scalar features (32)**

| Group            | Count | Features                                                                          |
|-----------------|-------|-----------------------------------------------------------------------------------|
| DOP              | 4     | mean, std, IQR, frac_dop_low                                                     |
| Step-angle       | 9     | mean, std, max, p99, kurtosis, skewness, burst_count, cum_arc, autocorr_lag1     |
| Theta-ref        | 4     | mean, std, range, p95                                                            |
| Stokes variance  | 6     | S1/S2/S3 std, S1/S2/S3 range                                                    |
| Welch PSD        | 9     | peak_freq, peak_power, bp_low, bp_mid, bp_high, ratio_mid_low, ratio_high_mid, spec_entropy, vb_snr_80hz |

**Frequency-domain features (1 001)**

Raw FFT magnitude bins `fft_bin_0` … `fft_bin_1000`  
(zero-padded FFT of the DOP-gated Stokes displacement signal, 1 Hz resolution)

### Train / test split strategy

Time-based per-file split with **no temporal leakage**:

```
for each CSV file:
    n_train = ceil(0.20 × n_rows)   # first 20 % of rows (chronologically earliest)
    train_rows = rows[0 : n_train]
    test_rows  = rows[n_train :]
```

### Models

| Model         | Key hyperparameters                                         |
|--------------|--------------------------------------------------------------|
| Random Forest | 100 trees, max_depth=15, class_weight='balanced'            |
| XGBoost       | 200 rounds, early_stopping=20, scale_pos_weight adapted     |
| SVM           | RBF kernel, C=10, class_weight='balanced', probability=True  |

All models use `StandardScaler` (fit on train, applied to test) and
`SimpleImputer(strategy='mean')` for missing values.

### Output directory structure

```
outputs_{SOURCE}_enhanced/
├── plots/
│   ├── 01_cm_random_forest.png      ← RF confusion matrix (normalised)
│   ├── 02_cm_xgboost.png            ← XGBoost confusion matrix
│   ├── 03_cm_svm.png                ← SVM confusion matrix
│   ├── 04_roc_auc_curves.png        ← ROC curves (macro average per model)
│   ├── 05_learning_curves.png       ← Rolling F1 score vs test window index
│   ├── 06_kmeans_clustering.png     ← K-Means (k=5) on PCA-projected test features
│   ├── 07_class_distribution.png    ← Train vs test class counts
│   ├── 08_model_comparison.png      ← Grouped bar: accuracy / precision / recall / F1
│   └── 09_feature_distributions.png ← Top-6 scalar features (box plots per class)
├── results.csv              ← Per-window predictions + class probabilities
├── model_performance.txt    ← Human-readable accuracy, F1, AUC, per-class report
├── feature_importance.csv   ← Top-50 features ranked by RF importance
├── training.log             ← Timestamped training log (DEBUG level)
└── config.yaml              ← Full run configuration (seed, params, data split sizes)
```

### Interpreting results

| Output file             | What to look at                                                    |
|-------------------------|---------------------------------------------------------------------|
| `model_performance.txt` | `F1 macro` and `AUC macro` give the headline per-model scores.     |
| `results.csv`           | Inspect `true_event` vs `pred_*` columns for per-window errors.    |
| `feature_importance.csv`| Top features guide future feature engineering or pruning.         |
| `plots/01–03`           | Dark diagonal → high accuracy; off-diagonal → class confusions.   |
| `plots/04`              | AUC close to 1.0 → good separability in probability space.        |
| `plots/06`              | Tight clusters per class → well-separated event signatures.        |

### Data quality checks performed automatically

- NaN count in S1/S2/S3 (logged as WARNING)
- Unit-sphere deviation (‖[S1,S2,S3]‖ − 1) exceeding 10⁻⁴ (logged as WARNING)
- Sampling-rate deviation > 500 Hz from nominal 2000 Hz (logged as WARNING)
- Missing or truncated files (< 1 MB) silently skipped

### Example results (10GE source)

```
Random Forest  acc=0.8148  F1=0.8081  AUC=0.6978  CV F1=0.9477±0.0114
XGBoost        acc=0.8739  F1=0.8645  AUC=0.9846  CV F1=0.9299±0.0305
SVM            acc=0.8168  F1=0.7787  AUC=0.6938  CV F1=0.8084±0.0193
```

### Reproducibility

Every run saves `config.yaml` recording the random seed (42), git commit hash,
data-split sizes, feature count, and model hyperparameters.  Re-running with
the same `config.yaml` settings will produce identical results.

---

## Feature Extraction Script (`analysis/feature_extraction.py`)

Standalone windowed feature extractor (25 features, 6-second windows, 50 % overlap)
used for exploratory analysis across all sources.

```bash
python analysis/feature_extraction.py
# Output: analysis/features_windowed.csv
```

---

## License

See repository for license information.