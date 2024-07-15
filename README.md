# :mushroom: Mushroom Genus Classification Fork

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)


## :star: Overall Goal
Deploy a machine learning model that classifies the genus of a mushroom based on its image.

### Background

Since we like to search for mushrooms in nature, this deployed model will help us learn more about the mushrooms that we see in the wild.

### Value Proposition

When we search for mushrooms in the wild, we are often clueless about the mushrooms that we actually see, because we are not so knowledgeable about mushrooms in general. Deploying a classifier would therefore make this process more enjoyable, since we would be able to identify a mushroom's genus in real-time.

## :star: Framework
We will use [`timm`](https://huggingface.co/docs/timm/index), a collection of PyTorch image models. As timm contains pretrained models, we will use them as starting points for the model that we eventually deploy. We plan to fine-tune them for our specific prediction task.

Furthermore, we will use [`PyTorch Lightning`](https://github.com/Lightning-AI/pytorch-lightning) to reduce boilerplate code and track our training experiments.

## :star: Dataset
### Data Introduction
The dataset chosen for this project is the :mushroom: [Mushroom Image dataset](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images) :mushroom: on Kaggle. It comprises 6714 images across 9 different mushroom genuses:
|  | Agaricus | Amanita | Boletus | Cortinarius | Entoloma | Hygrocybe | Lactarius | Russula | Suillus |
|-|----------|---------|---------|-------------|----------|-----------|-----------|---------|---------|
| Count | 353 | 750 | 1073 | 836 | 364 | 316 | 1563 | 1148 | 311 |

In case we choose to add more data or change the dataset, other options include:
- [Another dataset on Kaggle](https://www.kaggle.com/datasets/daniilonishchenko/mushrooms-images-classification-215), which contains 3122 images of 215 species of mushroom.

## :star: Models

### Types of Models

Our starting points will be
- [`Resnet50`](https://huggingface.co/timm/resnet50.a1_in1k), a pre-trained general purpose image classification model, since it [outperformed some other options on a related task](https://arxiv.org/pdf/2210.10351): poisonous mushroom classification.

- A CNN that we train ourselves on only the mushroom data, as a baseline.

Also, future iterations, and the eventually deployed model, are likely to be fine-tuned models from `timm`.

### Metrics

We will evaluate our models using classification accuracy, as it will feel more natural for potential users of our application who are not familiar with machine learning.

| Class      | True Positives | False Positives | True Negatives | False Negatives | Accuracy | Precision | Recall  | F1 Score |
|------------|----------------|-----------------|----------------|-----------------|----------|-----------|---------|----------|
| Boletus    | 52.0           | 14.0            | 836.0          | 109.0           | 0.878338 | 0.787879  | 0.322981| 0.458150 |
| Russula    | 92.0           | 133.0           | 705.0          | 81.0            | 0.788328 | 0.408889  | 0.531792| 0.462312 |
| Entoloma   | 29.0           | 170.0           | 786.0          | 26.0            | 0.806133 | 0.145729  | 0.527273| 0.228346 |
| Hygrocybe  | 23.0           | 7.0             | 956.0          | 25.0            | 0.968348 | 0.766667  | 0.479167| 0.589744 |
| Agaricus   | 6.0            | 16.0            | 942.0          | 47.0            | 0.937685 | 0.272727  | 0.113208| 0.160000 |

## :star: Rebuild

### How to install
Installing the project on your machine should be straighforward although Pytorch Geometric can cause some trouble. Clone the repo:
```bash
git clone https://github.com/cxzhang4/Mushroom_Classification.git
conda create -n myenv python=3.12
conda activate myenv
cd Mushroom_Classification
pip install -r requirements.txt
```

### How to run
Make sure data in located in data/raw with each class-named subfolder containing images.
You can use
```bash
make pull_data
```
Remember to change parameter values in Makefile and mushroom_classification/config/
```bash
make make_dataset
make train_model
make predict_model
make visualize
```
or use this to go through whole workflow
```bash
make run_pipeline
```
In order to use WandB you have to set "sweep" in *mushroom_classification/config/hydra* as *true*, then set environment variables:
```bash
export WANDB_API_KEY=***********************
export WANDB_PROJECT=***********
export WANDB_ENTITY=***********
```
## :star: Results
|  | Agaricus | Amanita | Boletus | Cortinarius | Entoloma | Hygrocybe | Lactarius | Russula | Suillus |
|-|----------|---------|---------|-------------|----------|-----------|-----------|---------|---------|
| Count | 353 | 750 | 1073 | 836 | 364 | 316 | 1563 | 1148 | 311 |

## :star: Project Structure
------------

    ├── Makefile                  <- Makefile with commands like `make data` etc.
    ├── README.md                 <- The top-level README for developers using this project.
    ├── docs                      <- A default Sphinx project; see sphinx-doc.org for details
    ├── models                    <- Save trained model.
    ├── mushroom_classification   <- Source code for use in this project.
    │   ├── __init__.py           <- Makes mushroom_classification a Python module.
    │   │
    │   ├── config                <- Config files.
    │   │   ├── hydra.yaml
    │   │   └── sweep.yaml
    │   │
    │   ├── fastapi_app           <- Fastapi files.
    │   │   ├── main.py
    │   │   ├── Dockerfile
    │   │   └── requirements.txt
    │   │
    │   ├── data                  <- Scripts to process data.
    │   │   ├── __init__.py
    │   │   └── make_dataset.py
    │   │
    │   ├── models                <- Scripts to train models and then use it to make predictions.
    │   │   ├── __init__.py
    │   │   ├── model.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization         <- Scripts to create visualization plots for predictions.
    │       ├── __init__.py
    │       └── visualize.py
    │
    ├── outputs                   <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── metrics.csv
    │   ├── predictions.npy
    │   └── true_labels.py
    │
    ├── reports                   <- Reports related.
    │   ├── figures
    │   │   └── *.png
    │   ├── README.md
    │   ├── *.html                <- Reports for coverage.
    │   └── report.py
    │
    ├── tests                     <- Scripts for tests.
    │   ├── __init__.py
    │   ├── test_make_dataset.py
    │   ├── test_model.py
    │   ├── test_predict_model.py
    │   ├── test_train_model.py
    │   └── test_visualize.py
    │
    ├── data.dvc          
    |
    ├── pyproject.toml
    │
    ├── requirements.txt           <- The requirements file for reproducing the model environment.
    │
    ├── requirements_test.txt      <- The requirements file for reproducing the analysis environment.
    |
    ├── mushroom.dockerfile 
    │
    └── setup.py                   <- makes project pip installable (pip install -e .) so src can be imported

--------


