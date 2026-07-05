# Dataset Installation

Datasets are described by JSON manifests and installed under `VIEWTRUST_DATA_ROOT`
or `./data`.

The example manifest is:

```text
configs/datasets.example.json
```

It uses placeholder URLs by default. Replace them with real dataset URLs and
checksums before downloading.

## First Priority 0 Dataset

The first recommended Priority 0 dataset path is the NeRF Synthetic chair mini
subset recipe:

```text
docs/NERF_SYNTHETIC_MINI.md
configs/nerf_synthetic_chair_minimal.yaml
scripts/data/prepare_nerf_synthetic_subset.py
```

This recipe assumes the raw NeRF Synthetic chair scene has already been
downloaded. It does not auto-download large datasets.

## Dry Run

Local-safe dry run:

```bash
python scripts/data/install_datasets.py \
  --manifest configs/datasets.example.json \
  --data-root ./data
```

This prints the install plan and does not download anything.

## Download

Server download:

```bash
python scripts/data/install_datasets.py \
  --manifest configs/datasets.example.json \
  --data-root "$VIEWTRUST_DATA_ROOT" \
  --download \
  --extract
```

Use real URLs and `sha256` values for reproducible installs. Dataset files,
archives, and extracted contents should not be committed.
