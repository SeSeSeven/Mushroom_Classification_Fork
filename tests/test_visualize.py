import pytest
import numpy as np
import pandas as pd
import os
import shutil
import torch
from PIL import Image
from pathlib import Path
from mushroom_classification import _PROJECT_ROOT, _PATH_DATA
from mushroom_classification.visualization.visualize import visualize_predictions

@pytest.fixture(scope="module")
def setup_directories():
    # Create a temporary directory structure
    processed_dir = Path(_PATH_DATA, 'processed_test')
    if not os.path.exists(processed_dir):
    # Create dummy data for testing
        for category in ["cat", "dog"]:
            category_path = Path(processed_dir, 'test', category)
            category_path.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                dummy_image = torch.rand(3, 300, 300) * 255
                fake_im = Image.fromarray(dummy_image.byte().permute(1, 2, 0).numpy())
                fake_im.save(Path(category_path, f'dummy_{i}.png'))
    
    output_dir = Path(_PROJECT_ROOT, 'outputs_test')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Create mock predictions
    if not os.path.exists(Path(output_dir, 'predictions.npy')):
        predictions = np.random.randint(0, 2, size=(4,))
        predictions = ['cat' if pred == 0 else 'dog' for pred in predictions]
        np.save(Path(output_dir, 'predictions.npy'), predictions)
    prediction_path = Path(output_dir, 'predictions.npy')
    
    # Create mock metrics
    if not os.path.exists(Path(output_dir, 'metrics.csv')):
        metrics_data = {
            'True Positive': [1, 2],
            'True Negative': [3, 2],
            'False Positive': [1, 0],
            'False Negative': [0, 1],
            'Accuracy': [0.9, 0.8],
            'Precision': [0.75, 0.85],
            'Recall': [0.7, 0.65],
            'F1 Score': [0.72, 0.74]
        }
        metrics_df = pd.DataFrame(metrics_data, index=['cat', 'dog'])
        metrics_df.to_csv(Path(output_dir, 'metrics.csv'), index=True)

    metrics_df_path = Path(output_dir, 'metrics.csv')
    
    report_dir = Path(_PROJECT_ROOT,'reports/figures_test')
    report_dir.mkdir(parents=True, exist_ok=True)
    # Return paths for use in tests
    yield processed_dir, prediction_path, metrics_df_path, report_dir

    # Clean up after tests
    shutil.rmtree(processed_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(report_dir, ignore_errors=True)

@pytest.mark.run(order=6)
def test_visualize_predictions(setup_directories):
    processed_dir, prediction_path, metrics_df_path, report_dir = setup_directories
    
    visualize_predictions(processed_dir, prediction_path, report_dir, 4, "(2,2)", 42, metrics_df_path)
    # Check if the output files are created
    assert os.path.exists(Path(report_dir, "predictions_visualization.png"))
    assert os.path.exists(Path(report_dir, "metrics.png"))

