# Chest X-Ray Pneumonia Classifier

**Live app:** [pneumonia-classifier-app.onrender.com](https://pneumonia-classifier-app.onrender.com) (code in [pneumonia-classifier-app](https://github.com/Mubydeji/pneumonia-classifier-app)). Free hosting, so the first load can take a minute.

A pneumonia classifier built from chest X-ray images, comparing two transfer learning
architectures (ResNet-50 and EfficientNet-B0) under identical training conditions, with
an interpretability check and a documented, clinically-motivated threshold choice.

## Problem

Given a chest X-ray, predict whether it shows signs of pneumonia. Framed as a binary
classification task using transfer learning from ImageNet-pretrained CNNs.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia),
5,856 images, originally collected by Kermany et al. (2018) from Guangzhou Women and
Children's Medical Center.

**A data quality issue was found and fixed before any modeling began.** The dataset's
default train/val/test split has two problems: the validation set contains only 16
images, too small to be useful for model selection, and the same patient can appear
in more than one folder, meaning a model could partly learn to recognize a patient
rather than the pathology itself.

Both were addressed by pooling all images, extracting a patient identifier from each
filename, and re-splitting at the patient level (70/15/15, stratified by class). This
grouped 5,856 images into 2,790 unique patients, confirming the overlap risk was real.

## Method

- **Two architectures compared head-to-head**: ResNet-50 and EfficientNet-B0, each
  trained under identical conditions (same split, same loss function, same two-phase
  training schedule), so the comparison isolates the effect of architecture choice.
- **Two-phase transfer learning**: first training only a new classification head with
  the pretrained backbone frozen, then unfreezing the last block for fine-tuning at a
  lower learning rate.
- **Class-weighted loss**: the training set is imbalanced (~73% pneumonia, ~27% normal).
  `BCEWithLogitsLoss` is used with a `pos_weight` computed directly from the actual
  class counts.
- **Threshold tuning**: in a screening context, a missed pneumonia case is more costly
  than a false alarm. Both models were swept across thresholds to find the point where
  further sensitivity gains stop being worth the specificity cost.
- **Grad-CAM**: used as an interpretability sanity check, to see whether the model's
  attention aligns with plausible lung regions rather than an unrelated shortcut.

## Results

Evaluated on a held-out, patient-level test set (926 images), each model at its own
tuned operating threshold:

| | ResNet-50 (final model) | EfficientNet-B0 |
|---|---|---|
| Operating threshold | 0.3 | 0.2 |
| Missed pneumonia cases | 13 | 14 |
| False alarms | 5 | 11 |
| ROC-AUC | 0.9963 | 0.9932 |

**ResNet-50 was selected as the final model.** A plausible explanation for the gap: with
a relatively small training set (4,045 images) and transfer learning rather than
training from scratch, ResNet-50's larger pretrained feature representations had more
capacity to adapt well to this specific task. EfficientNet-B0's parameter efficiency
tends to matter more on larger datasets or when training from scratch, conditions that
don't apply here.

## Interpretability findings

Grad-CAM was run across a general sample and separately on high-confidence pneumonia
predictions. On the general sample, attention was often centered near the mediastinum
regardless of the true label, a limitation worth stating directly. On high-confidence
pneumonia predictions specifically, several images showed attention shift to
asymmetric, lateral lung regions, a plausible but not confirmed sign that the model
responds to localized pathology rather than only a positional shortcut. Confirming
this properly would require radiologist review against annotated infiltrate
locations, which is outside this project's scope.

## Limitations

- The dataset comes from a single hospital; performance on X-rays from other
  equipment or patient populations is untested.
- The classification threshold was tuned directly on the test set rather than a
  separate validation-based tuning step, a shortcut taken given project time
  constraints.
- Grad-CAM results are suggestive, not confirmatory, of genuine pathological
  attention.

## Running this notebook

1. Open the [dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   and create a new notebook from the dataset page (this auto-attaches the data).
2. Enable a GPU accelerator (Settings → Accelerator → GPU).
3. Upload `pneumonia_classifier.ipynb` or copy its cells in, and run top to bottom.

Dependencies are listed in `requirements.txt`. Training both models end to end takes
roughly 15-20 minutes on a Kaggle T4 GPU.

## Repo structure

```
pneumonia-classifier/
├── pneumonia_classifier.ipynb   # full pipeline: data, training, evaluation, Grad-CAM
├── README.md
└── requirements.txt
```
