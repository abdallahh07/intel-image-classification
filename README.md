# Intel Image Classification

A CNN built in PyTorch that classifies natural-scene photos into 6 classes: buildings, forest, glacier, mountain, sea and street. The final model reaches **84.1% test accuracy** on 3,000 held-out images.

## Dataset

[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (Kaggle): 150×150 colour photos.

| Split | Images |
|---|---|
| Train | 14,034 |
| Test | 3,000 |

The classes are roughly balanced (about 2,200–2,500 training images each), so plain accuracy is a fair metric here. Training images are augmented with a random horizontal flip and a random rotation of up to 10°. The data is not in this repo; download it from Kaggle.

## Models compared

Three models were trained on the same data, from a weak baseline up to a CNN.

| Model | Description | Test accuracy |
|---|---|---|
| Model 0 | Linear layers only (no non-linearity), SGD | 48.5% |
| Model 1 | MLP with ReLU, SGD lr=0.1 | 17.5% |
| **Model 2** | **CNN: 3 blocks of 2 conv layers + max-pool, Adam lr=0.001, 15 epochs** | **84.1%** |

- **Model 0** shows that a model with no non-linearity (and no convolutions) can't solve the task.
- **Model 1** stayed at chance level (6 classes ≈ 16.7%). SGD with lr=0.1 was too aggressive for this network. I kept the result as a record of a failed run and did not tune it further.
- **Model 2** is the final model. Train accuracy ends at 85.3% and test accuracy at 84.1%. The small gap means the model is under-fitted, not over-fitted, so more epochs or capacity should improve it.

### CNN architecture

Input 3×150×150 → three blocks of [Conv(3×3) → ReLU → Conv(3×3) → ReLU → MaxPool(2)] with 20 channels → flatten (20×18×18) → Linear → 6 classes. About 6.5 minutes to train 15 epochs on a GPU.

## Error analysis

TODO: paste the confusion matrix image (`docs/confusion_matrix.png`) and the prediction grid (`docs/predictions.png`).

TODO: one or two sentences on which classes are confused (for example glacier vs mountain, buildings vs street) and why that makes sense visually.

## Limitations

- The test set was evaluated after every epoch to monitor training. The reported number is from the final epoch, not the best one, but there is no separate validation set.
- Single run, single seed, no cross-validation.
- Model 1's failure is a learning-rate problem and says nothing about MLPs in general.
- No pre-trained models were tried. Transfer learning (for example ResNet) would likely beat 84%.

## Run it

1. Download the dataset from Kaggle and place it in `data/intel_images/`.
2. Open `notebooks/intel_image_classification.ipynb` in Colab or Jupyter (a GPU is recommended).
3. Run all cells.

```bash
pip install torch torchvision matplotlib pandas tqdm
```
