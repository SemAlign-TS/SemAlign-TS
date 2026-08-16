# SemAlign-TS processed data

The processed data used for the main experiments are distributed separately because the archives are large.

**Google Drive:**
https://drive.google.com/drive/folders/1OcRrBivZ-EXpqR6BMG163IlBjfokDJB2?usp=sharing

The public folder contains six archives:

- `electricity.tar.gz`
- `etth1.tar.gz`
- `ettm1.tar.gz`
- `exchange_rate.tar.gz`
- `traffic.tar.gz`
- `weather.tar.gz`

It also contains `README.txt` and `SHA256SUMS.txt`.

Each dataset archive contains the main-experiment files used by SemAlign-TS:

- normalized length-96 windows (`*_X.npy`);
- textual conditions (`*_text.pkl`);
- frozen text embeddings and masks (`*_emb.npy`, `*_emb_mask.npy`);
- semantic / paired-window metadata (`*_meta.pkl`);
- dataset statistics and semantic thresholds (`*_stats.json`);
- official raw-observation-disjoint train/validation/test indices;
- excluded boundary-window indices.

## Local layout

Extract the six archives under `data/processed/` so the repository contains, for example:

```text
data/processed/electricity/electricity_X.npy
data/processed/etth1/etth1_X.npy
...
```

Example:

```bash
mkdir -p data/processed
tar -xzf electricity.tar.gz -C data/processed
tar -xzf etth1.tar.gz -C data/processed
tar -xzf ettm1.tar.gz -C data/processed
tar -xzf exchange_rate.tar.gz -C data/processed
tar -xzf traffic.tar.gz -C data/processed
tar -xzf weather.tar.gz -C data/processed
```

Paraphrase-specific embeddings are not part of the main data archive. They belong to the prompt-robustness analysis and may be regenerated separately.
