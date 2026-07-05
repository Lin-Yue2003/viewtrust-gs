# Dataset Installation

Datasets are described by JSON manifests and installed under `VIEWTRUST_DATA_ROOT`
or `./data`.

The example manifest is:

```text
configs/datasets.example.json
```

It uses placeholder URLs by default. Replace them with real dataset URLs and
checksums before downloading.

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
