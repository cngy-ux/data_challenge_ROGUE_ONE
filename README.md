# Data Challenge – Facial Occlusion Estimation

## Repository

This repository contains the different CNN-based approaches evaluated for the facial occlusion estimation challenge.

## Project Structure

### ResNet50 Experiments

| File                       | Description                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------------ |
| `Approche_CNN_MLP.py`      | Initial ResNet50 baseline trained for 10 epochs.                                                 |
| `Approche_CNN_MLP_v2.py`   | ResNet50 trained for 20 epochs.                                                                  |
| `Approche_CNN_MLP_v2_1.py` | ResNet50 with stronger data augmentation, cosine learning rate scheduler and 40 training epochs. |
| `Approche_CNN_MLP_v2_2.py` | ResNet50 with horizontal flip augmentation and cosine learning rate scheduler.                   |

### Alternative CNN Architectures

| File                     | Description                             |
| ------------------------ | --------------------------------------- |
| `Approche_CNN_MLP_v3.py` | EfficientNet-B3 based regression model. |
| `Approche_CNN_MLP_v4.py` | ConvNeXt-Tiny based regression model.   |

### Excluded Experiments

The repository also contains experiments using partially frozen backbones. These experiments were not retained because full fine-tuning consistently outperformed backbone freezing.

---

## Requirements

The experiments were developed and tested with:

* Python 3.11
* PyTorch 2.3+
* torchvision
* numpy
* pandas
* scikit-learn
* matplotlib
* tqdm

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## Expected Dataset Structure

The code expects the following directory structure:

```text
occlusion_datasets/
├── train.csv
├── test_students.csv

crops/
└── Crop_224_5fp_100K/
    ├── image_1.jpg
    ├── image_2.jpg
    └── ...
```

---

## Training Procedure

All models are initialized using ImageNet pre-trained weights and adapted to a regression task predicting a facial occlusion score between 0 and 1.

The final regression head consists of:

* Fully connected layer
* ReLU activation
* Dropout
* Fully connected output layer
* Sigmoid activation

Training uses:

* AdamW optimizer
* Weighted Mean Squared Error loss
* Batch size = 32

Some versions additionally use:

* Data augmentation
* CosineAnnealing learning rate scheduling

---

## Reproducing Results

### Local Training

Example:

```bash
python Approche_CNN_MLP_v2.py
```

or

```bash
python Approche_CNN_MLP_v2_2.py
```

depending on the experiment.

### Alternative Architectures

```bash
python Approche_CNN_MLP_v3.py
python Approche_CNN_MLP_v4.py
```

### Cluster Execution

Example Slurm script:

```bash
sbatch train_resnet.sh
```

The scripts automatically train the model, evaluate it on the validation and internal test sets, and generate predictions for leaderboard submission.

---

## Main Findings

The experiments show that:

* Full fine-tuning significantly outperforms freezing the backbone.
* ResNet50 provides competitive performance for facial occlusion estimation.
* Data augmentation must be carefully selected, as aggressive cropping can degrade performance.
* Learning rate scheduling improves training stability.
* CNN-based approaches remain strong baselines despite the popularity of recent Vision Transformer models.

---

## Best Model

The final submission was produced using the best-performing CNN configuration identified during experimentation.

Candidate models for the final submission include:

* `Approche_CNN_MLP_v2.py`
* `Approche_CNN_MLP_v2_2.py`

depending on the final leaderboard evaluation.

---

## Repository Link

This repository contains all code necessary to reproduce the experiments and final submission:

```text
https://github.com/cngy-ux/data_challenge_ROGUE_ONE
```

---

## Authors

**ROGUE ONE** – Data Challenge 2026
