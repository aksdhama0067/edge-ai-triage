# `data/`

## `dataset.csv` (already present)

This is **synthetic, mock data** generated for the hackathon demo — 25 fake
patient records with plausible-looking but entirely made-up values. It is
used nowhere in the training pipeline and contains no real images. It
exists only to illustrate the shape of clinical metadata the system is
designed around.

## Setting up a real training dataset

This project does **not** download or bundle any real dataset. To train the
lesion classifier, you need to supply your own, properly licensed image
dataset — for example a research-licensed release of
[HAM10000](https://doi.org/10.1038/sdata.2018.161) or
[ISIC](https://www.isic-archive.com/) data, or your own collected and
consented images. Check and comply with that dataset's license before use;
none of this project's code grants you rights to any dataset.

Once you have images, create:

```text
data/
  images/              <- your actual image files
    img_0001.jpg
    img_0002.jpg
    ...
  manifest.csv          <- see format below
```

### `manifest.csv` format

Required columns:

| column       | description                                                   |
|--------------|----------------------------------------------------------------|
| `image_path` | path to the image, relative to `data/images/` (or absolute)   |
| `label`      | one of: `Melanoma`, `Basal Cell Carcinoma`, `Benign Keratosis` |
| `patient_id` | **Strongly recommended** stable patient identifier for patient-level splitting; never expose this in the public UI |

Additional columns may be kept for analysis. `patient_id` is special: when it
is present for every row, the training script uses patient-level train/validation/test
splitting to reduce leakage from multiple images belonging to the same person.

Example:

```csv
image_path,label,patient_id
img_0001.jpg,Melanoma,P001
img_0002.jpg,Benign Keratosis,P002
img_0003.jpg,Basal Cell Carcinoma,P003
```

Then train with:

```bash
cd ../training
python train.py --manifest ../data/manifest.csv --data-root ../data/images --output ../models
```

See `../models/README.md` for what happens to the resulting weights, and
`../training/train.py --help` for all training options (epochs, batch size,
learning rate, split fractions, early-stopping patience).

## A note on class balance and dataset size

The training script applies class-balanced sampling and augmentation, but
neither substitutes for enough real, diverse, well-labeled data. A model
trained on a small or narrow dataset will not generalize — treat
`"model_mode": "trained"` as "the weights file exists," not as a claim
about real-world accuracy. Actual performance should always be checked
against `training/evaluate.py`'s held-out test metrics before trusting any
number this app reports.
