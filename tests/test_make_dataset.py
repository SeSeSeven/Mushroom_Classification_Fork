import os
import shutil
import pytest
import torch
from pathlib import Path
from PIL import Image
from tests import _PATH_DATA, _PROJECT_ROOT
from mushroom_classification.data.make_dataset import app

from typer.testing import CliRunner

runner = CliRunner()

@pytest.mark.run(order=1)
def test_data_path_exists():
    """Test if the data path exists."""
    assert os.path.exists(Path(_PATH_DATA, 'raw')) or os.path.exists(Path(_PROJECT_ROOT, 'data.dvc')), f"Directory does not exist: {Path(_PATH_DATA, 'raw')}"

@pytest.fixture(scope="function")
def setup_directories():
    raw_dir = Path(_PATH_DATA,'raw_test')
    for category in ["cat", "dog"]:
        category_path = Path(raw_dir, category)
        category_path.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            dummy_image = torch.rand(3, 300, 300) * 255
            fake_im = Image.fromarray(dummy_image.byte().permute(1, 2, 0).numpy())
            fake_im.save(Path(category_path, f'dummy_{i}.png'))

    processed_dir = Path(_PATH_DATA, 'processed_test')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    yield raw_dir, processed_dir

    # Clean up after test
    shutil.rmtree(raw_dir, ignore_errors=True)
    # shutil.rmtree(raw_dir, ignore_errors=True) # leave for following tests

@pytest.mark.run(order=2)
def test_split_data(setup_directories):
    raw_dir, processed_dir = setup_directories

    result = runner.invoke(app, [
        "--raw_dir", raw_dir,
        "--processed_dir", processed_dir,
        "--val_size", 0.2,
        "--test_size", 0.2,
        "--random_state", 42
    ])

    assert result.exit_code == 0

    # Check if the directories have been created
    for split in ["train", "validation", "test"]:
        for category in ["cat", "dog"]:
            assert Path(processed_dir, split, category).exists()
            assert len(list(Path(processed_dir, split, category).glob("*"))) > 0
