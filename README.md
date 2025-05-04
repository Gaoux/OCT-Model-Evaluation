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

### 📦 Data Sources

This project uses publicly available labeled OCT image datasets:

1. **[Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images (V3)](https://data.mendeley.com/datasets/rscbjbr9sj/3)**  
   Kermany D, Goldbaum M, Cai W et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell. 2018; 172(5):1122-1131. [doi:10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)

2. **[Labeled OCT and Chest X-Ray Images for Classification (V2)](https://data.mendeley.com/datasets/rscbjbr9sj/2)**  
   Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018). *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data, V2. [doi:10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2)

---

### 🧠 Notebooks & Models

This project uses or adapts code/models from the following Kaggle notebooks, released under the **Apache 2.0 open-source license**:

1. **[Retinal OCT Feature Map and Filters Visualization](https://www.kaggle.com/code/justforgags/retinal-oct-feature-map-and-filters-visualization/notebook)**  
2. **[96.1% in Retinal OCT CNN Model](https://www.kaggle.com/code/mohamedgobara/96-1-in-retinal-oct-cnn-model/notebook)**

---

### 📜 License & Usage

This repository is intended solely for **educational and research purposes**.  
Please ensure compliance with the licensing terms of the respective datasets and models.
 



