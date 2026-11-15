# ASPAN EON: CNN Classification

This is the machine-learning part of ASPAN EON, Team The K'Oirat's SPHEREx webtool. We trained a small convolutional neural network (CNN) to predict an object's catalog category from its FITS images. The 3 categories are **moving**, **stable**, and **variable**. The model predicts a category from image features; it does not directly track motion or measure changes over time

The reason for a low amount of git commits comes from the developer not considering creating a GitHub Repository for the Machine Learning part of this project, at the start of the Hackathon (NASA SPACE APPS CHALLENGE, 2026)

## Files

| File | Purpose |
|---|---|
| `CNN_Step_by_Step.ipynb` | Data preparation, training, and evaluation in Jupyter |
| `predict.py` | Load a saved model and classify FITS cutouts |
| `export_website_predictions.py` | Export predictions for the website's prepared objects |
| `training_runs/20261114_154707_390720/best_cnn.pt` | Selected trained model |
| `training_runs/20261114_154707_390720/training.json` | Object splits, settings, and training results |

Keep the checkpoint and its `training.json` together

## Setup

Install the Python packages:

```bash
python -m pip install numpy astropy matplotlib scikit-learn torch jupyter
```

Place the dataset archives in a folder named `nasa_3` next to the notebook. It expects one `catalogs*.zip` containing `catalogs/metadata.csv`, plus image archives named `raw*.zip` or `fits*.zip`. The notebook reads them directly, so they do not need to be extracted for training
The full FITS dataset is distributed separately and is not included in this ML repository

## Training

Open `CNN_Step_by_Step.ipynb` in Jupyter and run the cells in order. The notebook reads the images, normalizes their brightness, pads them to a common size, and adds a mask identifying valid pixels
Training uses a **70/15/15 split by object**, **20 epochs**, and **pool size 2**. All observations of an object stay in the same split so the model is evaluated on different objects from those used for training

## Predicting a new image

Run this from the repository folder, replacing `example.fits` with your file:

```bash
python predict.py training_runs/20261114_154707_390720/best_cnn.pt example.fits
```

The saved model expects a 2-dimensional FITS `IMAGE` no larger than 84×84 pixels. The command returns the predicted category and scores for all three classes -> You can pass several FITS files after the checkpoint path if they belong to the same object

## Results

The selected run reached an object-level validation macro F1 of 0.767. In the reported test, it correctly classified **11 of 15 objects**, about so about 73% accuracy -> This is a small test set, so the real potential is yet to be seen. Model scores are not proof that an object moves or changes brightness

## Credits

Built by The K' Oirat for the NASA Space Apps Challenge 2026

Trained using SPHEREx observations from the NASA/IPAC Infrared Science Archive (IRSA), collected and prepared by our team

The CNN was implemented with PyTorch, with Astropy used to read FITS files

- [SPHEREx mission](https://science.nasa.gov/mission/spherex/)
- [IRSA archive](https://irsa.ipac.caltech.edu/Missions/spherex.html)