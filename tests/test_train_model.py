import pytest
from pathlib import Path
from mushroom_classification import _PATH_DATA
from mushroom_classification.models.train_model import train_model

@pytest.mark.run(order=4)
def test_train_model():
    """Test the train_model function with given parameters."""
    processed_dir = Path(_PATH_DATA, 'processed_test')
    save_model = Path(_PATH_DATA, 'models_test', 'resnet50.pt')

    # 确保 train_model 使用正确的参数
    try:
        train_model(
            vertex_ai=False,  # Assuming default is local, change as needed
            processed_dir=str(processed_dir), 
            save_model=str(save_model)
        )
    except SystemExit as e:
        assert e.code == 0, f"train_model exited with a non-zero code: {e.code}"

    # Check if the model has been saved
    assert save_model.exists(), f"Model file does not exist: {save_model}"
