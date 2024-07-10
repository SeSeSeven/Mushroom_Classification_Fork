import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.cm as cm
from typing import Tuple
from torchvision import datasets, transforms
import random
from PIL import UnidentifiedImageError
from pathlib import Path
from mushroom_classification import _PROJECT_ROOT, _PATH_DATA
import typer
from typing_extensions import Annotated

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        """Override the __getitem__ method to skip corrupted images."""
        try:
            return super(SafeImageFolder, self).__getitem__(index)
        except (UnidentifiedImageError, OSError):
            print(f"Skipping corrupted image at index {index}")
            new_index = (index + 1) % len(self)
            return self.__getitem__(new_index)

def parse_tuple(s: str) -> Tuple[int, int]:
    """Parse a string of the form '(int, int)' into a tuple of integers."""
    s = s.strip('()')
    parts = s.split(',')
    if len(parts) != 2:
        raise ValueError
    return (int(parts[0]), int(parts[1]))

app = typer.Typer()
@app.command()
def visualize_predictions(data_dir: Annotated[str,typer.Option("--processed_dir",'-p')]=str(Path(_PATH_DATA,'processed')), 
                          prediction_path: Annotated[str,typer.Option("--prediction_path",'-e')]=str(Path(_PROJECT_ROOT,'outputs/predictions.npy')),
                          report_dir: Annotated[str,typer.Option("--report_dir",'-o')]=str(Path(_PROJECT_ROOT,'reports/figures')), 
                          num_images: Annotated[int,typer.Option("--num_images",'-i')]=16,
                          figure_arrage: Annotated[str,typer.Option("--figure_arrage",'-f')]="(4,4)",
                          random_state: Annotated[int,typer.Option("--random_state",'-r')]=42,
                          metrics_path: Annotated[str,typer.Option("--metrics_path",'-m')]=str(Path(_PROJECT_ROOT,'outputs/metrics.csv'))) -> None:
    """Visualize predictions on a subset of test images and metrics for each class using bar charts."""
    transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
    ])
    
    test_dataset = SafeImageFolder(root=Path(data_dir,'test'), transform=transform)
    class_names = test_dataset.classes

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    fig = plt.figure(figsize=(15, 15))
    predictions = np.load(prediction_path)
    num_images = min(num_images, len(test_dataset), len(predictions))
    random.seed(random_state)
    indices = random.sample(range(len(test_dataset)), num_images)

    plot_r, plot_c = parse_tuple(figure_arrage)
    for i, idx in enumerate(indices):
        image, label = test_dataset[idx]
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        
        ax = fig.add_subplot(plot_r, plot_c, i+1, xticks=[], yticks=[])
        ax.imshow(image)
        pred_label = predictions[idx] if idx < len(predictions) else 'Unknown'
        ax.set_title(f"True: {class_names[label]} \nPred: {pred_label}")
    
    output_path = Path(report_dir, 'predictions_visualization.png')
    plt.savefig(output_path)
    plt.close(fig)
    typer.echo(f"Predictions visualization saved to {output_path}")

    # Load the metrics_df
    metrics_df = pd.read_csv(metrics_path, index_col=0)
    # Plot each metric
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    # Define the colormap
    colormap = cm.get_cmap('magma', metrics_df.shape[0])  # 9 is the number of classes
    colors = [colormap(i) for i in range(colormap.N)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, metric in enumerate(metric_names):
        axes[i].bar(metrics_df.index, metrics_df[metric], color=colors)
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel('')
        axes[i].set_title(f'{metric} per Class')
        axes[i].set_xticks(metrics_df.index)
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    # Save plot to reports directory
    plt.savefig(Path(report_dir, 'metrics.png'))
    plt.close(fig)
    typer.echo(f"Metrics visualization saved to {report_dir}")

if __name__ == "__main__":
    app()
