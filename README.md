# Digit Recognizer High-Score CNN

This repository contains a strong TensorFlow/Keras solution for the Kaggle
[Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition.

The code trains a convolutional neural network with image augmentation, learning
rate scheduling, early stopping, model checkpointing, ensembling, and test-time
augmentation. It reads the standard Kaggle CSV files and writes a submission CSV
with `ImageId` and `Label`.

## Project Structure

```text
DigitRecognizer/
├── train_high_score.py
├── README.md
├── .gitignore
├── train/
│   └── train.csv
└── test/
    └── test.csv
```

The data files are intentionally ignored by Git because they are large and
should be downloaded from Kaggle or kept locally.

## Environment

Python 3.10 or newer is recommended.

Install the required packages:

```powershell
py -m pip install numpy pandas scikit-learn tensorflow
```

On native Windows, TensorFlow 2.11+ uses CPU by default. For faster training,
run the project in WSL2 with a GPU-enabled TensorFlow setup.

## Data

Place the Kaggle files in these paths:

```text
train/train.csv
test/test.csv
```

`train.csv` must contain the `label` column followed by `pixel0` through
`pixel783`. `test.csv` must contain only `pixel0` through `pixel783`.

## Quick Pipeline Check

Use this to verify the full training and prediction pipeline on a small subset:

```powershell
py .\train_high_score.py --quick-check --epochs 1 --models 1 --tta 1 --output test\submission_quick_check.csv
```

This is only a smoke test. Do not submit the quick-check output to Kaggle.

## Train a Strong Baseline

```powershell
py .\train_high_score.py
```

Default settings:

- 3 CNN models
- 30 epochs per model
- batch size 128
- 10% stratified validation split
- 4 rounds of test-time augmentation

The submission is saved to:

```text
test/submission_high_score.csv
```

## Higher-Score Run

For a slower but stronger ensemble:

```powershell
py .\train_high_score.py --models 5 --epochs 40 --tta 8
```

For an even more aggressive run:

```powershell
py .\train_high_score.py --models 10 --epochs 45 --tta 12
```

The best settings depend on available CPU/GPU time. On CPU-only Windows, start
with the default command first.

## Output

The script creates a Kaggle-compatible CSV:

```text
ImageId,Label
1,2
2,0
3,9
...
```

Upload `test/submission_high_score.csv` to Kaggle for scoring.

## Notes

- Model checkpoints are saved under `models/` during training and ignored by
  Git.
- The script uses different random seeds for each model in the ensemble.
- Test-time augmentation averages predictions from the original test images and
  several lightly augmented versions.
