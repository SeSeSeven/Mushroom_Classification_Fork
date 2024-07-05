import torch
import numpy as np
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError

from models.model import MushroomClassifier, get_model
import pytorch_lightning as pl

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super(MyImageFolder, self).__getitem__(index)
        except (UnidentifiedImageError, OSError) as e:
            print(f"Skipping corrupted image at index {index}: {e}")
            new_index = (index + 1) % len(self)
            return self.__getitem__(new_index)

def predict(data_dir, model_path, model_name, num_classes, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
    ])
    
    test_dataset = MyImageFolder(root=f"{data_dir}/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = MushroomClassifier(model_name=model_name, num_classes=num_classes)
    model = MushroomClassifier.load_from_checkpoint(model_path, model_name=model_name, num_classes=num_classes)
    model.eval()
    
    # Print the shape of a batch of test data
    for images, labels in test_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        break  # Just print the first batch for debugging purposes
    
    all_predictions = []
    all_probabilities = []
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu', devices=1 if torch.cuda.is_available() else "auto")
    
    # Define a custom predict_step to handle the batch correctly
    class CustomPredictModel(MushroomClassifier):
        def predict_step(self, batch, batch_idx):
            images, _ = batch
            return self(images)

    # Load the model into CustomPredictModel
    custom_model = CustomPredictModel.load_from_checkpoint(
        checkpoint_path=model_path,
        model_name=model_name,
        num_classes=num_classes
    )

    predictions = trainer.predict(custom_model, dataloaders=test_loader)
    for batch_preds in predictions:
        probabilities = torch.softmax(batch_preds.cpu(), dim=1)
        _, predicted = torch.max(probabilities, 1)
        all_predictions.extend(predicted.numpy())
        all_probabilities.extend(probabilities.numpy())
    return all_predictions, all_probabilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Main function for splitting dataset.",
    )

    parser.add_argument(
        "data_dir", 
        type=str, 
        help="Path to splitted dataset directory."
    )

    parser.add_argument(
        "model_path", 
        type=str, 
        help="Path to the trained model."
    )

    parser.add_argument(
        "--num_classes",
        "-c",
        default=9,
        type=int,
        help="Class number of dataset.",
    )

    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="resnet50.a1_in1k",
        help="Model name for training from timm.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    model_path = args.model_path
    num_classes = args.num_classes
    model_name = "hf_hub:timm/" + args.model_name
    predictions, probabilities = predict(data_dir, model_path, model_name, num_classes)
    np.save("/predictions.npy", predictions)
    np.save("/prob.npy", probabilities)
    print(predictions)
