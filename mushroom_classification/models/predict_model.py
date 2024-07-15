import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import UnidentifiedImageError
from mushroom_classification import _PROJECT_ROOT
from model import MushroomClassifier
import pytorch_lightning as pl
import typer
from typing_extensions import Annotated
import hydra
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict

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
def predict_model(vertex_ai: Annotated[str, typer.Option("--vertex-ai", "-v")] = None,
                  processed_dir: Annotated[str, typer.Option("--processed_dir", '-p')] = None, 
                  save_model: Annotated[str, typer.Option("--save_model", '-s')] = None, 
                  output_dir: Annotated[str, typer.Option("--output_dir", '-o')] = None) -> None:
    """Predict the class of mushrooms using a trained model, and calculate metrics from the prediction results."""
    
    if vertex_ai == "true":
        config_name = "hydra_vertex"
    elif vertex_ai == "false":
        config_name = "hydra_local"
    else:
        raise ValueError("Invalid value for --vertex-ai. Use 'true' or 'false'.")
    
    @hydra.main(version_base=None, config_path=_PROJECT_ROOT/'mushroom_classification/config', config_name=config_name)
    def inner_predict(cfg: DictConfig) -> None:
        # Merge command line arguments with Hydra config
        if processed_dir:
            cfg.data.processed_dir = processed_dir
        if save_model:
            cfg.data.model_save_path = save_model
        if output_dir:
            cfg.data.output_dir = output_dir

        print(OmegaConf.to_yaml(cfg))

        transform = transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.ToTensor(),
        ])
        
        test_dataset = SafeImageFolder(root=Path(cfg.data.processed_dir, 'test'), transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False)
        class_names = test_dataset.classes

        model = MushroomClassifier(model_name=cfg.model.model_name, num_classes=cfg.model.num_classes)
        model = MushroomClassifier.load_from_checkpoint(cfg.data.model_save_path, model_name=cfg.model.model_name, num_classes=cfg.model.num_classes)
        model.eval()
        
        if not os.path.exists(cfg.data.output_dir):
            os.makedirs(cfg.data.output_dir)

        all_predictions = []
        trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1)
        
        # Define a custom predict_step to handle the batch correctly
        class CustomPredictModel(MushroomClassifier):
            """Custom predict model that overrides the predict_step method."""
            def predict_step(self, batch):
                images, labels = batch
                return self(images), labels

        # Load the model into CustomPredictModel
        custom_model = CustomPredictModel.load_from_checkpoint(
            checkpoint_path=cfg.data.model_save_path,
            model_name=cfg.model.model_name,
            num_classes=cfg.model.num_classes
        )

        true_labels = []
        predictions = trainer.predict(custom_model, dataloaders=test_loader)
        if predictions is None:
            typer.echo("No predictions were made.")
            raise ValueError
        for batch_preds, batch_labels in predictions:
            probabilities = torch.softmax(batch_preds.cpu(), dim=1)
            _, predicted = torch.max(probabilities, 1)
            all_predictions.extend(predicted.numpy())
            true_labels.extend(batch_labels.cpu().numpy())
        
        unique_labels = np.unique(true_labels)
        label_to_class = {label: class_names[label] for label in unique_labels}
        all_predictions = pd.Series(all_predictions).map(label_to_class).tolist()
        true_labels = pd.Series(true_labels).map(label_to_class).tolist()

        np.save(Path(cfg.data.output_dir, 'predictions.npy'), all_predictions)
        typer.echo(f"Predictions saved to {Path(cfg.data.output_dir, 'predictions.npy')}")
        np.save(Path(cfg.data.output_dir, 'true_labels.npy'), true_labels)
        typer.echo(f"True labels saved to {Path(cfg.data.output_dir, 'true_labels.npy')}")

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
        metrics_df.to_csv(Path(cfg.data.output_dir, 'metrics.csv'), index=True)
        typer.echo(f"Metrics saved to {Path(cfg.data.output_dir, 'metrics.csv')}")

    inner_predict()

if __name__ == "__main__":
    app()
