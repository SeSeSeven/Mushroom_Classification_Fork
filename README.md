# Mushroom Species Classification

## Overall Goal
Deploy a machine learning model capable of differentiating between various species of mushrooms based on images.

## Framework
We will learn and improve based on [timm](https://huggingface.co/docs/timm/index), a collection of PyTorch image models, to obtain a suitable model for our specific classification task.

## Dataset
### Data Introduction
The dataset chosen for this project is the :mushroom: [Mushroom Image dataset](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images) from Kaggle. It comprises 6714 images across 9 different mushroom species:
- **Agaricus**
- **Amanita**
- **Boletus**
- **Cortinarius**
- **Entoloma**
- **Hygrocybe**
- **Lactarius**
- **Russula**
- **Suillus**

### Data Split
The dataset will be divided into training and testing sets using a 4:1 split ratio.

## Models


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
