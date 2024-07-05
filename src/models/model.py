import timm
import torch
from torch import nn
import pytorch_lightning as pl

class MushroomClassifier(pl.LightningModule):
    def __init__(self, model_name="hf_hub:timm/resnet50.a1_in1k", num_classes=9, lr=0.001):
        super(MushroomClassifier, self).__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=True)
        in_features = self.model.get_classifier().in_features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
    
    def forward(self, x):
        print(f"Input shape to forward: {x.shape}")
        x = self.model.forward_features(x)
        print(f"Shape after forward_features: {x.shape}")
        x = self.global_pool(x)  # Apply global pooling: output shape will be [batch_size, 2048, 1, 1]
        print(f"Shape after global_pool: {x.shape}")
        x = torch.flatten(x, 1)   # Flatten to [batch_size, 2048]
        print(f"Shape after flatten: {x.shape}")
        x = self.classifier(x)   # Classifier layer: input shape [batch_size, 2048], output shape [batch_size, num_classes]
        print(f"Output shape from forward: {x.shape}")
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def get_model(model_name="hf_hub:timm/resnet50.a1_in1k", num_classes=9, lr=0.001):
    return MushroomClassifier(model_name, num_classes, lr)
