# Retinal OCT Model Testing

## Overview

This repository is dedicated to testing pre-trained deep learning models for Retinal OCT (Optical Coherence Tomography) classification. The goal is to evaluate the model's performance on a given dataset and analyze its predictions through various metrics and visualizations.

We do not train the model in this repository; we solely use and assess it.

## Dataset

The dataset used for testing consists of labeled OCT images. Images are organized into folders corresponding to their respective categories, such as:

- `CNV/`
- `DME/`
- `DRUSEN/`
- `NORMAL/`

Each folder contains test images for the respective class.

## Installation

To set up the required environment, use the following:

**Using pip:**

```sh
pip install -r requirements.txt
```

**Using Conda:**

```sh
conda env create -f environment.yml
conda activate oct-testing
```

## Credits

This project uses a pre-trained model from the following source:
[96.1% in Retinal OCT CNN Model](https://www.kaggle.com/code/mohamedgobara/96-1-in-retinal-oct-cnn-model)

## License

This project is for educational and research purposes only.
