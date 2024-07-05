import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import random
import sys
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, '/content/drive/MyDrive/mushroom_classfication')
from src.models.model import get_model, MushroomClassifier

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super(MyImageFolder, self).__getitem__(index)
        except (UnidentifiedImageError, OSError):
            print(f"Skipping corrupted image at index {index}")
            new_index = (index + 1) % len(self)
            return self.__getitem__(new_index)

def visualize_training_data(data_dir, output_dir='reports/figures', num_images=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_dataset = MyImageFolder(root=f"{data_dir}/train", transform=transform)
    class_names = train_dataset.classes

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure(figsize=(15, 15))
    for i in range(min(num_images, len(train_dataset))):
        image, label = train_dataset[random.randint(0, len(train_dataset) - 1)]
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        ax.imshow(image)
        ax.set_title(f"Label: {class_names[label]}")
    
    output_path = os.path.join(output_dir, 'training_data_visualization.png')
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Training data visualization saved to {output_path}")

def visualize_predictions(data_dir, predictions, output_dir='reports/figures', num_images=16):
    transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
    ])
    
    test_dataset = MyImageFolder(root=f"{data_dir}/test", transform=transform)
    class_names = test_dataset.classes

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure(figsize=(15, 15))
    num_images = min(num_images, len(test_dataset), len(predictions))
    indices = random.sample(range(len(test_dataset)), num_images)

    for i, idx in enumerate(indices):
        image, label = test_dataset[idx]
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        ax.imshow(image)
        pred_idx = predictions[idx] if idx < len(predictions) else -1
        pred_label = class_names[pred_idx] if pred_idx >= 0 and pred_idx < len(class_names) else 'Unknown'
        ax.set_title(f"True: {class_names[label]} \nPred: {pred_label}")
    
    output_path = os.path.join(output_dir, 'predictions_visualization.png')
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Predictions visualization saved to {output_path}")

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, output_dir='reports/figures'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Val Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Training curves saved to {output_path}")

def visualize_feature_maps(model, data_dir, output_dir='reports/figures', num_images=4):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
    ])
    
    test_dataset = MyImageFolder(root=f"{data_dir}/test", transform=transform)
    
    # Register hook on a specific layer (e.g., the last convolutional layer)
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Find the specific layer to hook (e.g., a specific convolutional layer)
    target_layer = model.model.layer4[2].conv3  # Example: last convolutional layer in layer4 block
    
    handle = target_layer.register_forward_hook(get_activation('conv3'))
    
    images, _ = zip(*[test_dataset[i] for i in range(num_images)])
    images = torch.stack(images)
    
    with torch.no_grad():
        _ = model(images)
    
    handle.remove()
    
    feature_maps = activation['conv3']
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(num_images, min(8, feature_maps.size(1)), figsize=(30, 30))
    for i in range(num_images):
        for j in range(min(8, feature_maps.size(1))):
            axes[i, j].imshow(feature_maps[i, j].cpu().numpy(), cmap='viridis')
            axes[i, j].axis('off')
    
    output_path = os.path.join(output_dir, 'feature_maps.png')
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Feature maps saved to {output_path}")

if __name__ == "__main__":
    data_dir = "data"
    
    train_losses = [0.9, 0.8, 0.7, 0.6]
    val_losses = [1.0, 0.9, 0.8, 0.75]
    train_accuracies = [0.6, 0.65, 0.7, 0.75]
    val_accuracies = [0.55, 0.6, 0.65, 0.7]
    
    predictions = [random.randint(0, 8) for _ in range(100)]  # Dummy predictions for example

    model = get_model()
    
    visualize_training_data(data_dir)
    visualize_predictions(data_dir, predictions)
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    visualize_feature_maps(model, data_dir)
