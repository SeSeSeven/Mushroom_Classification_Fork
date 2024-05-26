# Mushroom Species Classification

## Overall Goal
Deploy a machine learning model capable of differentiating between various genera (genuses) of mushrooms based on images.

Since we like to search for mushrooms in nature, this deployed model will help us learn more about the mushrooms that we see in the wild.

## Framework
We will learn and improve based on [timm](https://huggingface.co/docs/timm/index), a collection of PyTorch image models, to obtain a suitable model for our specific classification task.

As timm contains pretrained models, we will use them as starting points. We plan to fine-tune them for our specific prediction task.

Furthermore, we will use [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) to reduce boilerplate code and track our training experiments.

## Dataset
### Data Introduction
The dataset chosen for this project is the :mushroom: [Mushroom Image dataset](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images) on Kaggle. It comprises 6714 images across 9 different mushroom genuses:
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
- [Another dataset on Kaggle](https://www.kaggle.com/datasets/daniilonishchenko/mushrooms-images-classification-215)

### Data Split
The dataset will be divided into training and testing sets using a 4:1 split ratio.

## Models
Our starting points will be
- [Resnet50](https://huggingface.co/timm/resnet50.a1_in1k), a pre-trained general purpose image classification model, since it outperformed some other options on a related task: https://arxiv.org/pdf/2210.10351 
- A CNN that we train ourselves on only the mushroom data

## Experiments
Our experimental setup will include:
1. **Data Augmentation**: Applying transformations like rotation, flipping, and scaling to increase the robustness of the model.
2. **Model Fine-Tuning**: Fine-tuning pre-trained models from `timm` on our mushroom dataset.
3. **Hyperparameter Tuning**: Using techniques like grid search or random search to find the best hyperparameters for our models.
4. **Cross-Validation**: Implementing k-fold cross-validation to ensure the reliability and generalizability of our models.
5. **...**

## Results
The results section will include:
- **Accuracy**: The overall accuracy of each model on the test set.
- **Visualization**: Visual representation of the classification performance.
- **Final Conclusion**: Our summary and findings on classification results.
- **...**

## Source
The source code for this project is available in the `src` directory. It includes:
- Data preprocessing scripts
- Model training scripts
- Evaluation scripts
- Deployment scripts
- ...

## Notes
- **Requirements**: A `requirements.txt` file is included for easy installation of necessary Python packages.
- **Reproducibility**: All experiments will be logged and tracked using tools like MLflow to ensure reproducibility. Additionally, we will create configuration files for our dataset and hyperparameters to maintain consistency, facilitate data management, and implement version control.
- **...**
