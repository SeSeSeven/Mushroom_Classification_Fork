# :mushroom: Mushroom Genus Classification

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

## :star: Rebuild

### Reproduce using the newest build (Docker image):
The newest build of the repo is provided as an docker image stored on google cloud. Image can be pulled from the Google Cloud Container with the following command:
```bash
docker pull gcr.io/linear-rig-337909/group5_proj_cpu_container:latest
```
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
pull_data
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

    ├── Makefile                 <- Makefile with commands like `make data` etc.
    ├── README.md                <- The top-level README for developers using this project.
    ├── docs                     <- A default Sphinx project; see sphinx-doc.org for details
    ├── models                   <- Save trained model.
    ├── mushroom_classification  <- Source code for use in this project.
    │   ├── __init__.py          <- Makes mushroom_classification a Python module.
    │   │
    │   ├── config               <- Config files.
    │   │   ├── hydra.yaml
    │   │   └── sweep.yaml
    │   │
    │   ├── data                 <- Scripts to process data.
    │   │   ├── __init__.py
    │   │   └── make_dataset.py
    │   │
    │   ├── models               <- Scripts to train models and then use it to make predictions.
    │   │   ├── __init__.py
    │   │   ├── model.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization         <- Scripts to create visualization plots for predictions.
    │       ├── __init__.py
    │       └── visualize.py
    ├── tests                     <- Scripts for unitests.
    │   ├── __init__.py
    │   ├── test_make_dataset.py
    │   ├── test_model.py
    │   ├── test_predict_model.py
    │   ├── test_train_model.py
    │   └── test_visualize.py
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    |
    ├── pyproject.toml         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

