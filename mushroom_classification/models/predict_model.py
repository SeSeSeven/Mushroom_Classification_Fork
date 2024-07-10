import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import UnidentifiedImageError
from mushroom_classification import _PROJECT_ROOT, _PATH_DATA
from mushroom_classification.models.model import MushroomClassifier
from typing import Tuple, Union
from collections import defaultdict
import pytorch_lightning as pl
import typer
from typing_extensions import Annotated

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        """Override the __getitem__ method to skip corrupted images."""
        try:
            return super(SafeImageFolder, self).__getitem__(index)
        except (UnidentifiedImageError, OSError) as e:
            print(f"Skipping corrupted image at index {index}: {e}")
            new_index = (index + 1) % len(self)
            return self.__getitem__(new_index)

app = typer.Typer()
@app.command()
def predict(data_dir: Annotated[str,typer.Option("--processed_dir",'-p')]=str(Path(_PATH_DATA,'processed')), 
            save_model: Annotated[str,typer.Option("--save_model",'-s')]=str(Path(_PROJECT_ROOT,'models','resnet50.ckpt')), 
            model_name: Annotated[str,typer.Option("--model_name",'-n')]="resnet50.a1_in1k", 
            num_classes: Annotated[int,typer.Option("--num_classes",'-c')]=9, 
            batch_size: Annotated[int,typer.Option("--batch_size",'-b')]=32,
            output_dir: Annotated[str,typer.Option("--output_dir",'-o')]=str(Path(_PROJECT_ROOT,'outputs'))) -> Tuple[Union[np.ndarray, list], Union[np.ndarray, list], pd.DataFrame]:
    """Predict the class of mushrooms using a trained model, and calculate metrics from the prediction results."""
    transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
    ])
    
    test_dataset = SafeImageFolder(root=Path(data_dir,'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    class_names = test_dataset.classes

    model = MushroomClassifier(model_name=model_name, num_classes=num_classes)
    model = MushroomClassifier.load_from_checkpoint(save_model, model_name=model_name, num_classes=num_classes)
    model.eval()
    
    # Print the shape of a batch of test data
    for images, labels in test_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        break  # Just print the first batch for debugging purposes
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_predictions = []
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu', devices=1 if torch.cuda.is_available() else "auto")
    
    # Define a custom predict_step to handle the batch correctly
    class CustomPredictModel(MushroomClassifier):
        """Custom predict model that overrides the predict_step method."""
        def predict_step(self, batch):
            images, labels = batch
            return self(images), labels

    # Load the model into CustomPredictModel
    custom_model = CustomPredictModel.load_from_checkpoint(
        checkpoint_path=save_model,
        model_name=model_name,
        num_classes=num_classes
    )

    true_labels = []
    predictions = trainer.predict(custom_model, dataloaders=test_loader)
    if predictions is None:
        typer.echo("No predictions were made.")
        raise ValueError
    for batch_preds , batch_labels in predictions:
        probabilities = torch.softmax(batch_preds.cpu(), dim=1)
        _, predicted = torch.max(probabilities, 1)
        all_predictions.extend(predicted.numpy())
        true_labels.extend(batch_labels.cpu().numpy())
    
    unique_labels = np.unique(true_labels)
    label_to_class = {label: class_names[label] for label in unique_labels}
    all_predictions = pd.Series(all_predictions).map(label_to_class).tolist()
    true_labels = pd.Series(true_labels).map(label_to_class).tolist()

    np.save(Path(output_dir,'predictions.npy'), all_predictions)
    typer.echo(f"Predictions saved to {Path(output_dir,'predictions.npy')}")
    np.save(Path(output_dir,'true_labels.npy'), true_labels)
    typer.echo(f"True labels saved to {Path(output_dir,'true_labels.npy')}")

    # Initialize dictionaries to hold counts
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})

    # Process each prediction
    for pre, trlab in zip(all_predictions, true_labels):
        # Update counts for true positives, false positives, true negatives, and false negatives
        for label in set(all_predictions):
            if pre == label and trlab == label:
                class_metrics[label]['tp'] += 1
            elif pre == label and trlab != label:
                class_metrics[label]['fp'] += 1
            elif pre != label and trlab == label:
                class_metrics[label]['fn'] += 1
            else:
                class_metrics[label]['tn'] += 1

    # Calculate metrics for each class
    metrics = {}
    for label, counts in class_metrics.items():
        tp = counts['tp']
        fp = counts['fp']
        tn = counts['tn']
        fn = counts['fn']
        
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        metrics[label] = {
            'True Positives': tp,
            'False Positives': fp,
            'True Negatives': tn,
            'False Negatives': fn,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        }
    
    metrics_df = pd.DataFrame(metrics).T
    # Also save metrics as a CSV file
    metrics_df.to_csv(Path(output_dir, 'metrics.csv'), index=True)
    typer.echo(f"Metrics saved to {Path(output_dir,'metrics.csv')}")

    return all_predictions, true_labels, metrics_df

if __name__ == "__main__":
    predictions, true_labels, metrics_df = app()
    print(metrics_df)

