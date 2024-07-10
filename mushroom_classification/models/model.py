import timm
from typing import Tuple
import torch
from torch import nn
import pytorch_lightning as pl

class MushroomClassifier(pl.LightningModule):
    """Builds a Classifier using models from timm for mushroom classification.

    Arguments:
        model_name: str, name of the model to use from timm
        num_classes: int, number of classes in the dataset
        lr: float, learning rate for the optimizer

    """

    def __init__(self, model_name: str = "resnet50.a1_in1k", 
                 num_classes: int = 9, lr: float=0.001) -> None:
        super(MushroomClassifier, self).__init__()
        self.save_hyperparameters()
        self.model = timm.create_model('hf_hub:timm/'+model_name, pretrained=True)
        in_features = self.model.get_classifier().in_features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network, returns the output."""
        x = self.model.forward_features(x)
        x = self.global_pool(x)  # Apply global pooling: output shape will be [batch_size, 2048, 1, 1]
        x = torch.flatten(x, 1)  # Flatten to [batch_size, 2048]
        x = self.classifier(x)   # Classifier layer: input shape [batch_size, 2048], output shape [batch_size, num_classes]
        return x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Training step for the model."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Validation step for the model."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def get_model(model_name: str="resnet50.a1_in1k", num_classes: int=9, 
              lr: float=0.001) -> MushroomClassifier:
    """Returns an instance of MushroomClassifier."""
    return MushroomClassifier(model_name, num_classes, lr)
