import torch
import pytest
import shutil
import os
from PIL import Image
from pathlib import Path
from mushroom_classification import _PROJECT_ROOT, _PATH_DATA
from mushroom_classification.models.model import MushroomClassifier
from mushroom_classification.models.predict_model import predict

@pytest.fixture(scope="module")
def setup_directories():
    """Fixture to set up directories and create dummy data for testing."""
    processed_dir = Path(_PATH_DATA, 'processed_test')
    if not processed_dir.exists():
        for category in ["cat", "dog"]:
            category_path = processed_dir / 'test' / category
            category_path.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                dummy_image = torch.rand(3, 300, 300) * 255
                fake_im = Image.fromarray(dummy_image.byte().permute(1, 2, 0).numpy())
                fake_im.save(category_path / f'dummy_{i}.png')

    model_dir = Path(_PROJECT_ROOT, 'models_test')
    dummy_model_path = model_dir / 'test_model.ckpt'
    if not dummy_model_path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(MushroomClassifier().state_dict(), dummy_model_path)

    outputs_dir = Path(_PROJECT_ROOT, 'outputs_test')
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    yield processed_dir, dummy_model_path, outputs_dir

    # Clean up after tests
    shutil.rmtree(processed_dir, ignore_errors=True)
    shutil.rmtree(model_dir, ignore_errors=True)
    shutil.rmtree(outputs_dir, ignore_errors=True)

@pytest.mark.run(order=5)
def test_predict(setup_directories):
    """Test the predict function with the set up directories."""
    processed_dir, dummy_model_path, outputs_dir = setup_directories

    # 调整参数数量以匹配 predict 函数的定义
    predictions, true_labels, metrics_df = predict(
        vertex_ai=False,  # Assuming default is local, change as needed
        processed_dir=str(processed_dir), 
        save_model=str(dummy_model_path), 
        output_dir=str(outputs_dir)
    )

    # Check if the output files have been created
    assert (outputs_dir / 'predictions.npy').exists()
    assert (outputs_dir / 'true_labels.npy').exists()
    assert (outputs_dir / 'metrics.csv').exists()

    assert len(predictions) > 0
    assert len(predictions) == len(true_labels)
