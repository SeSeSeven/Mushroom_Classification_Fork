import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from mushroom_classification.models.model import MushroomClassifier, get_model

# Define the test function
@pytest.mark.run(order=3)
def test_mushroom_classifier():
    # Create a model instance
    model = get_model()
    
    # Check if model is an instance of MushroomClassifier
    assert isinstance(model, MushroomClassifier), "Model is not an instance of MushroomClassifier"
    
    # Create dummy data
    inputs = torch.randn(2, 3, 224, 224)  # Batch of 2, 3 color channels, 224x224 image size
    labels = torch.randint(0, 9, (2,))  # Batch of 2, with random labels from 0 to 8
    
    # Test the forward pass
    outputs = model(inputs)
    assert outputs.shape == (2, 9), "Output shape is incorrect"
    
    # Create a dummy data loader
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Test the training step
    for batch in dataloader:
        loss = model.training_step(batch)
        assert isinstance(loss, torch.Tensor), "Training step did not return a tensor"
    
    # Test the validation step
    for batch in dataloader:
        loss = model.validation_step(batch)
        assert isinstance(loss, torch.Tensor), "Validation step did not return a tensor"
    
    # Test the optimizer configuration
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer), "Optimizer is not an instance of torch.optim.Adam"

