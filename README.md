# Mushroom Genus Classification

## Overall Goal
Deploy a machine learning model that classifies the genus of a mushroom based on its image.

### Background

Since we like to search for mushrooms in nature, this deployed model will help us learn more about the mushrooms that we see in the wild.

### Value Proposition

When we search for mushrooms in the wild, we are often clueless about the mushrooms that we actually see, because we are not so knowledgeable about mushrooms in general. Deploying a classifier would therefore make this process more enjoyable, since we would be able to identify a mushroom's genus in real-time.

## Framework
We will use [timm](https://huggingface.co/docs/timm/index), a collection of PyTorch image models. As timm contains pretrained models, we will use them as starting points for the model that we eventually deploy. We plan to fine-tune them for our specific prediction task.

Furthermore, we will use [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) to reduce boilerplate code and track our training experiments.

## Dataset
### Data Introduction
The dataset chosen for this project is the :mushroom: [Mushroom Image dataset](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images) :mushroom: on Kaggle. It comprises 6714 images across 9 different mushroom genuses:
- **Agaricus**
- **Amanita**
- **Boletus**
- **Cortinarius**
- **Entoloma**
- **Hygrocybe**
- **Lactarius**
- **Russula**
- **Suillus**

In case we choose to add more data or change the dataset, other options include:
- [Another dataset on Kaggle](https://www.kaggle.com/datasets/daniilonishchenko/mushrooms-images-classification-215), which contains 3122 images of 215 species of mushroom.

## Models

### Types of Models

Our starting points will be
- [Resnet50](https://huggingface.co/timm/resnet50.a1_in1k), a pre-trained general purpose image classification model, since it [outperformed some other options on a related task](https://arxiv.org/pdf/2210.10351): poisonous mushroom classification.
- A CNN that we train ourselves on only the mushroom data, as a baseline.

### Metrics

We will evaluate our models using classification accuracy, as it will feel more natural for potential users of our application who are not familiar with machine learning.
